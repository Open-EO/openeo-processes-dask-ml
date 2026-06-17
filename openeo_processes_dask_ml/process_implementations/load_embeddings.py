import json
from datetime import datetime as Datetime

import pystac
import rioxarray
import shapely
import xarray as xr
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


def _load_parquet(path: str) -> xr.DataArray:
    pass


def _load_embedding_item(stac_item: pystac.Item, asset_name: str) -> xr.DataArray:
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
        "vnd.apache.parquet"
    ):
        emb_cube = _load_parquet(path)
        return emb_cube

    # if parquet: load_parquet
    # if zarr: load_zarr
    raise TypeError(f"Loading embeddings of media-type {media_type} unsupported")


def _load_embedding_collection() -> xr.DataArray:
    pass


def load_embeddings(
    url: str,
    spatial_extent: list[float] = None,
    temporal_extent: list[float] = None,
    asset_name: str = "embeddings",
) -> xr.DataArray:
    stac_obj_dict = stac_utils.load_stac_json(url)

    # todo: validate STAC and embedding extension

    try:
        stac_obj = pystac.read_dict(stac_obj_dict)
    except pystac.STACTypeError as e:
        raise OpenEOException("Provided URL does not point to a valid STAC object")

    if isinstance(stac_obj, pystac.Item):
        return _load_embedding_item(stac_obj, asset_name)
    elif isinstance(stac_obj, pystac.Collection):
        return _load_embedding_collection()

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
