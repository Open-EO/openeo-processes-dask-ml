import json
from datetime import datetime as Datetime

import dask
import dask_geopandas
import fsspec
import geopandas as gpd
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import pyproj
import pystac
import pystac_client
import rioxarray
import shapely
import xarray as xr
import xvec
from dask import array as da
from dask.delayed import delayed
from numpy.typing import NDArray
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval
from openeo_processes_dask.process_implementations.data_model import VectorCube
from openeo_processes_dask.process_implementations.exceptions import OpenEOException

from openeo_processes_dask_ml.process_implementations.utils import stac_utils


def _get_item_time(stac_item: pystac.Item) -> Datetime:
    if isinstance(stac_item.datetime, Datetime):
        return stac_item.datetime

    dt_start = stac_item.properties.get("start_datetime")
    if isinstance(dt_start, str):
        start = Datetime.fromisoformat(dt_start)
        return start

    # in theory this should never occur as it would violate the STAC spec
    raise ValueError("Could not determine the item's datetime")


def _match_geom_in_list(
    geometry_list: list[shapely.geometry.base.BaseGeometry],
    geometry: shapely.geometry.base.BaseGeometry,
    tolerance: float,
) -> int | None:
    """
    Check weather a geometry is already present in a list of geometries, considering a
    coordinate matching tolerance to accomate for float rounding errors
    :param geometry_list: List of geometries to check
    :param geometry: the polygon to check
    :param tolerance: Tolerance to check
    :return: index of polygon in list, None if polygon is not present in list
    """
    for i, p in enumerate(geometry_list):
        if p.equals_exact(geometry, tolerance=tolerance):
            return i
    return None


def load_zarr():
    # if url ends with .zip: download to cache, unzip
    # check number of data variables: if 1: load this, if more: check if one is called "embeddings"

    # if it does not end with zip: load remotely from zarr store

    pass


def _load_tiff(path: str) -> xr.DataArray:
    dc = rioxarray.open_rasterio(path, chunks=True)
    dc = dc.rename({"band": "embedding"})  # rename bands dim to embedding

    # squeeze x and y if its only 1 (i.e. tiff only has one pixel
    if len(dc.coords["x"]) == 1 and len(dc.coords["y"]) == 1:
        dc = dc.squeeze(dim=["x", "y"], drop=True)

    dc.attrs.clear()  # remove rioxarray stats
    return dc


def _prepare_geoparquet(
    path: str,
    bbox: BoundingBox | None,
    geom_column_name: str,
    emb_column_name: str,
    emb_size: int,
    emb_dtype: np.dtype,
) -> tuple[NDArray[shapely.Geometry], da.Array]:
    @delayed
    def _stack_partition(series):
        # series is a pandas Series of numpy arrays -> 2D array
        return np.stack(series.to_numpy())

    # todo: comvert both to wgs84 if it is not yet
    gdf = dask_geopandas.read_parquet(path, columns=[geom_column_name, emb_column_name])

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox.west, bbox.south, bbox.east, bbox.north
        bbox_geom = shapely.box(xmin, ymin, xmax, ymax)
        gdf_bbox = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
        gdf = dask_geopandas.sjoin(gdf, gdf_bbox, how="inner", predicate="intersects")

    # this does the trick of stacking the arrays, but is VERY slow
    # dask checks if the arrays are of same length and can be stacked -> SLOW
    # embedding_array = da.stack(gdf_embeddings[emb_column_name])

    # same effect but more efficient: we already KNOW that each array is same length
    # 1) make one delayed dask object per partition
    # 2) for each delayed dask object make a delayed (lazy) dask array
    # 3) concat all lazy dask arrays to one big lazy dask array
    geom_parts = gdf[
        geom_column_name
    ].to_delayed()  # delayed objects, one per partition, for both columns
    emb_parts = gdf[emb_column_name].to_delayed()

    # compute geometries per partition (you need them as numpy coords anyway).
    # This ALSO gives us the exact row count of every partition.
    geoms_by_part = dask.compute(*geom_parts)  # list of pandas Series
    lengths = [len(g) for g in geoms_by_part]  # known rows per partition
    geoms_array = np.concatenate([g.to_numpy() for g in geoms_by_part])

    arrays = [
        da.from_delayed(
            _stack_partition(part),
            shape=(length, emb_size),  # rows per partition unknown -> nan
            dtype=emb_dtype,
        )
        for part, length in zip(emb_parts, lengths)
    ]
    embedding_array = da.concatenate(arrays, axis=0)
    return geoms_array, embedding_array


def _load_parquet_item(path: str, bbox: BoundingBox) -> xr.DataArray:
    # check geom column
    # check embedding column (embedding or embeddings?)
    # check if col is correct dtype (list? what float?)
    with fsspec.open(path, "rb") as file:
        parquet_schema = pq.read_schema(file)

    # search for embedding column
    col_names = [n.name for n in parquet_schema]
    possible_embedding_col_names = ["embedding", "embeddings", "emb", "embs"]
    for pos_emb_col_name in possible_embedding_col_names:
        if pos_emb_col_name in col_names:
            emb_column_name = pos_emb_col_name
            break
    else:
        raise Exception(
            f"Could not identify embedding column. Must be named one of "
            f"{','.join(possible_embedding_col_names)}"
        )

    # get embedding column info: embedding length and datatype
    emb_col_dtype = parquet_schema.field(emb_column_name).type
    if isinstance(emb_col_dtype, pyarrow.FixedSizeListType):
        emb_size = emb_col_dtype.list_size
        emb_dtype = emb_col_dtype.value_type.to_pandas_dtype()
    else:
        raise NotImplementedError("not supported yet")

    # get geometry column name
    geo_metadata_bytes = parquet_schema.metadata.get(b"geo")
    if geo_metadata_bytes:
        # Parse JSON bytes
        geo_metadata = json.loads(geo_metadata_bytes.decode("utf-8"))

        # Get the primary geometry column name
        geom_column_name = geo_metadata.get("primary_column")
    else:
        # not a valid geoparquet, let's try to find a geom column anyway
        pot_geom_col_names = ["geometry", "geom"]
        for p in pot_geom_col_names:
            if p in col_names:
                geom_column_name = p
                break
        else:
            raise Exception(
                f"Coule not identify geometry column. Provide a valid GeoParquet file "
                f"or name the geometry column one of {','.join(pot_geom_col_names)}"
            )

    geom_coords, emb_values = _prepare_geoparquet(
        path, bbox, geom_column_name, emb_column_name, emb_size, emb_dtype
    )
    emb_cube = xr.DataArray(
        emb_values, dims=["geometry", "embs"], coords={"geometry": geom_coords}
    ).xvec.set_geom_indexes("geometry", crs="EPSG:4326")

    emb_values.visualize("out.png")

    return emb_cube


def _load_embedding_item(
    stac_item: pystac.Item, asset_name: str, bbox: BoundingBox | None
) -> xr.DataArray:
    embedding_asset = stac_item.assets[asset_name]
    media_type = embedding_asset.media_type
    path = embedding_asset.href

    # we assume that embeddings as tif or parquet are purely spatial
    # if its it zarr, it can be spatial or spatio-temporal

    # load geotif file
    if media_type.startswith("image/tif"):
        footprint = shapely.from_geojson(json.dumps(stac_item.geometry))
        time = _get_item_time(stac_item)
        emb_cube = _load_tiff(path)
        emb_cube = emb_cube.expand_dims({"geometry": [footprint], "time": [time]})

        return emb_cube

    # if parquet file:
    if media_type.startswith("application/x-parquet") or media_type.startswith(
        "application/vnd.apache.parquet"
    ):
        emb_cube = _load_parquet_item(path, bbox=bbox)
        return emb_cube

    # if parquet: load_parquet
    # if zarr: load_zarr
    raise TypeError(f"Loading embeddings of media-type {media_type} unsupported")


def _load_embedding_collection_tif(items: pystac.ItemCollection, asset_name: str):
    item_arrays: list[list[xr.DataArray]] = []
    time_coords = []
    geom_coords = []

    for stac_item in items:
        path = stac_item.assets[asset_name].href
        footprint = shapely.from_geojson(json.dumps(stac_item.geometry)).normalize()
        time = _get_item_time(stac_item)
        emb_cube = _load_tiff(path)

        if time not in time_coords:
            time_coords.append(time)
            time_index = len(time_coords) - 1
            item_arrays.append(len(geom_coords) * [xr.DataArray()])
        else:
            time_index = time_coords.index(time)

        footprint_index = _match_geom_in_list(geom_coords, footprint, tolerance=0.00001)
        if footprint_index is None:
            geom_coords.append(footprint)
            footprint_index = len(geom_coords) - 1
            for time_step in item_arrays:
                time_step.append(xr.DataArray())

        item_arrays[time_index][footprint_index] = emb_cube

    single_temp_cubes = []
    for single_timestep_arrays in item_arrays:
        x = xr.concat(single_timestep_arrays, dim="geometry")
        single_temp_cubes.append(x)

    embedding_cube = xr.concat(single_temp_cubes, dim="time")
    embedding_cube = embedding_cube.assign_coords(
        {"geometry": geom_coords, "time": time_coords}
    )
    embedding_cube = embedding_cube.xvec.set_geom_indexes("geometry", crs="EPSG:4326")
    return embedding_cube


def load_embedding_collection_parquet():
    pass


# parts of this function have been taken from this script
# https://github.com/Open-EO/openeo-processes-dask/blob/main/openeo_processes_dask/process_implementations/cubes/load.py
def _load_embedding_collection(
    url: str,
    spatial_extent: BoundingBox | None = None,
    temporal_extent: TemporalInterval | None = None,
    asset_name: str = "embeddings",
) -> xr.DataArray:
    collection = pystac.read_file(url)

    catalog_url, collection_id = stac_utils.search_for_parent_catalog(url)
    query_params = {"collections": [collection_id]}
    stac_client = pystac_client.Client.open(catalog_url)

    # parse temporal extent
    if temporal_extent is not None:
        start_date = (
            str(temporal_extent[0].to_numpy())
            if temporal_extent[0] is not None
            else None
        )
        end_date = (
            str(temporal_extent[1].to_numpy())
            if temporal_extent[1] is not None
            else None
        )
        query_params["datetime"] = [start_date, end_date]

    # parse spatial extent
    if spatial_extent is not None:
        try:
            spatial_extent_4326 = spatial_extent
            if spatial_extent.crs is not None and not pyproj.crs.CRS(
                spatial_extent.crs
            ).equals("EPSG:4326"):
                raise Exception(
                    "Currently, only a bounding box in in EPSG:4326 is supported"
                )
            bbox = [
                spatial_extent_4326.west,
                spatial_extent_4326.south,
                spatial_extent_4326.east,
                spatial_extent_4326.north,
            ]
            query_params["bbox"] = bbox
        except Exception as e:
            raise Exception(f"Unable to parse the provided spatial extent: {e}")

    items = stac_client.search(**query_params).item_collection()

    if len(items) == 0:
        raise Exception(
            "Could not find any embeddings in the STAC collection for the provided "
            "spatial bounding box and/or timespan."
        )

    # figure out the data format that all assets have in common
    if collection.item_assets:
        if asset_name not in collection.item_assets:
            raise Exception
        emb_data_format = collection.item_assets[asset_name].media_type
    else:
        # parse from all items
        # i = items[0].assets["a"].media_type
        emb_data_formats = [
            i.assets[asset_name].media_type
            for i in items
            if i.assets.get(asset_name) is not None
        ]
        all_same_media_type = all(emb_data_formats[0] == e for e in emb_data_formats)

        if not all_same_media_type:
            raise Exception

        emb_data_format = emb_data_formats[0]

    # use media-type specific loader
    if emb_data_format.startswith("image/tif"):
        embedding_cube = _load_embedding_collection_tif(items, asset_name)
        return embedding_cube
    else:
        raise NotImplementedError(f"Cannot read embeddings of type {emb_data_format}")


def load_embeddings(
    url: str,
    spatial_extent: BoundingBox | None = None,
    temporal_extent: TemporalInterval | None = None,
    asset_name: str = "embeddings",
) -> xr.DataArray:
    stac_obj_dict = stac_utils.load_stac_json(url)

    # todo: validate STAC and embedding extension

    try:
        stac_obj = pystac.read_dict(stac_obj_dict)
    except pystac.STACTypeError as e:
        raise OpenEOException("Provided URL does not point to a valid STAC object")

    if isinstance(stac_obj, pystac.Item):
        return _load_embedding_item(stac_obj, asset_name, spatial_extent)
    elif isinstance(stac_obj, pystac.Collection):
        return _load_embedding_collection(
            url, spatial_extent, temporal_extent, asset_name
        )
    raise NotImplementedError(
        f"Loading of a STAC object of type {stac_obj.STAC_OBJECT_TYPE} is not supported"
    )

    # if item:
    # 1) check media type: if zarr: load with zarr, if geotiff: laod with rasterio (?), if gpq load as geoparquet

    # if collection: filter by bbox and temporal
    # checkl item-asset media type: if raster (e.g. zarr, geotiff) raise not implemented for now (?)
    # if geoparquet: load all of them, sort out spatial and temporal to form datacube
    # then cut (again) by bbox and temp

    # reproject into a harmonized grid (wgs84)
    # rename dimensions to harmonize everything: x,y,time,embedding
