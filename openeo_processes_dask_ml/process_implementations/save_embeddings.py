import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import dask.dataframe as ddf
import dask_geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from dask.delayed import Delayed, delayed
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.constants import (
    OPENEO_RESULTS_PATH,
)
from openeo_processes_dask_ml.process_implementations.utils import dim_utils, zip_utils


def _get_stac_item_template(_id: str) -> dict:
    d = {
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/embeddings/v0.0.1/schema.json"
        ],
        "type": "Feature",
        "id": _id,
        "collection": None,
        "links": [{"rel": "self", "href": f"./{_id}.json"}],
        "bbox": None,  # will be set later,
        "geometry": None,  # will be set later,
        "properties": {
            "datetime": None,
            "start_datetime": None,
            "end_datetime": None,
            # "gsd": None,
            "title": "EO-Embeddings",
            "description": "EO embeddings produced using openeo-processes-dask-ml",
            "emb:type": None,  # will be set later
            "emb:dimensions": None,  # will be set later
            "emb:chip_layout": {"layout_type": None},
            "data_type": None,  # will be set later
        },
        "assets": {
            "embeddings": {
                "href": None,
                "title": "embeddings",
                "type": None,
                "roles": ["embedding"],
            }
        },
    }
    return d


def _save_as_zarr(datacube: xr.DataArray, result_dir: Path, zarr_dir: Path) -> Path:
    saved = datacube.to_zarr(zarr_dir)
    zip_path = zip_utils.create_zip_archive(
        result_dir, zarr_dir, "results.zarr.zip", saved
    )
    return zip_path


def _set_stac_spatial_metadata_raster(stac_metadata: dict, datacube: xr.DataArray):
    x_dim, y_dim = dim_utils.get_spatial_dim_names(datacube)

    # todo: convert coords to wgs84
    xmin = float(min(datacube.coords[x_dim].data))
    ymin = float(min(datacube.coords[y_dim].data))
    xmax = float(max(datacube.coords[x_dim].data))
    ymax = float(max(datacube.coords[y_dim].data))
    bbox = [xmin, ymin, xmax, ymax]

    geom = {
        "type": "Polygon",
        "coordinates": [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ],
    }

    stac_metadata["bbox"] = bbox
    stac_metadata["geometry"] = geom


def _set_stac_time_metadata(stac_metadata: dict, datacube: xr.DataArray):
    try:
        time_dim = dim_utils.get_time_dim_name(datacube)
        if len(datacube.coords[time_dim]) > 1:
            tmin = min(datacube.coords[time_dim].data)
            tmax = max(datacube.coords[time_dim].data)
            stac_metadata["properties"]["start_datetime"] = str(tmin)
            stac_metadata["properties"]["end_datetime"] = str(tmax)
            del stac_metadata["properties"]["datetime"]
        else:
            t = datacube.coords[time_dim].data[0]
            stac_metadata["properties"]["datetime"] = str(t)
    except DimensionMissing:
        dt = str(datetime.now())
        stac_metadata["properties"]["datetime"] = dt
        del stac_metadata["properties"]["start_datetime"]
        del stac_metadata["properties"]["end_datetime"]


def _set_stac_embedding_metadata(stac_metadata: dict, datacube: xr.DataArray):
    emb_dim = dim_utils.get_embedding_dim_name(datacube)
    stac_metadata["properties"]["emb:type"] = "patch"
    stac_metadata["properties"]["emb:dimensions"] = len(datacube.coords[emb_dim].data)
    stac_metadata["properties"]["data_type"] = str(datacube.dtype)


def _set_stac_embedding_metadata_raster(stac_metadata: dict, datacube: xr.DataArray):
    _set_stac_embedding_metadata(stac_metadata, datacube)
    stac_metadata["properties"]["emb:chip_layout"]["layout_type"] = "regular_grid"


def _set_stac_embedding_asset_metadata_raster(
    stac_metadata: dict, out_path: Path
) -> dict:
    stac_metadata["assets"]["embeddings"]["href"] = str(out_path.absolute())
    stac_metadata["assets"]["embeddings"]["type"] = "application/vnd.zarr"
    return stac_metadata


def _update_stac_metadata_raster_cube(
    stac_metadata: dict, datacube: xr.DataArray, out_path: Path
):
    _set_stac_spatial_metadata_raster(stac_metadata, datacube)
    _set_stac_time_metadata(stac_metadata, datacube)
    _set_stac_embedding_metadata_raster(stac_metadata, datacube)


def _get_crs(da: xr.DataArray, geom_dim: str):
    try:
        crs = da.xvec.crs
        return crs.get(geom_dim) if isinstance(crs, dict) else crs
    except Exception:
        return da[geom_dim].attrs.get("crs")


def _column_labels(da, time_dim, time_fmt) -> list[str]:
    if time_dim not in da.dims:
        return ["embedding"]
    labels = []
    for t in da[time_dim].values:  # coords are always eager
        ts = pd.Timestamp(t)
        labels.append(
            f"embedding_{ts.strftime(time_fmt) if time_fmt else ts.isoformat()}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate embedding column names — use a finer `time_fmt`.")
    return labels


def _to_column(block: np.ndarray, arrow_dtype: pd.ArrowDtype | None):
    """(n_rows, n_emb) ndarray -> pandas column of per-row vectors."""
    block = np.ascontiguousarray(block)
    if arrow_dtype is None:  # object dtype fallback
        return list(block)
    fsl = pa.FixedSizeListArray.from_arrays(pa.array(block.reshape(-1)), block.shape[1])
    return pd.arrays.ArrowExtensionArray(fsl)


def _partition_to_gdf(blocks, geoms, columns, crs, index, arrow_dtype):
    """Runs inside one dask task: one geometry chunk, all timestamps."""
    data = {c: _to_column(b, arrow_dtype) for c, b in zip(columns, blocks)}
    gdf = gpd.GeoDataFrame(
        data, geometry=gpd.GeoSeries(geoms, crs=crs, index=index), index=index
    )
    return gdf[["geometry", *columns]]


def _meta(columns, crs, arrow_dtype) -> gpd.GeoDataFrame:
    dtype = arrow_dtype if arrow_dtype is not None else object
    meta = gpd.GeoDataFrame(
        {c: pd.Series([], dtype=dtype) for c in columns},
        geometry=gpd.GeoSeries([], crs=crs),
    )
    return meta[["geometry", *columns]]


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def _vector_cube_to_gdf(
    da: xr.DataArray,
    path: Path,
    geom_dim: str = "geometry",
    emb_dim: str = "embedding",
    time_dim: str = "time",
    time_fmt: str | None = None,
    geom_chunk: int = 50_000,  # used only if the cube isn't dask-backed yet
    arrow: bool = True,  # fixed_size_list columns instead of object dtype
):
    """
    Vector data cube (xvec) -> lazy ``dask_geopandas.GeoDataFrame``.

    Columns: ``geometry`` + ``embedding`` (no time dim) or one
    ``embedding_{iso_date}`` column per timestamp. Nothing is read until
    ``.compute()`` / ``.to_parquet()``.

    The time axis is *sliced*, not rechunked: each timestamp is an independent
    column, so no shuffle along time is required.
    """

    for dim in (geom_dim, emb_dim):
        if dim not in da.dims:
            raise ValueError(f"Missing required dimension {dim!r}; dims={da.dims}")
    extra = set(da.dims) - {geom_dim, emb_dim, time_dim}
    if extra:
        raise ValueError(f"Unexpected extra dimension(s): {sorted(extra)}")

    has_time = time_dim is not None
    order = (geom_dim, time_dim, emb_dim) if has_time else (geom_dim, emb_dim)
    da = da.transpose(*order)

    # we shouldnt need this as cube is dask-backed
    # if da.chunks is None:
    #     da = da.chunk({geom_dim: geom_chunk})

    # only the embedding axis must be contiguous per row; time is left alone
    if len(da.chunks[da.get_axis_num(emb_dim)]) > 1:
        da = da.chunk({emb_dim: -1})

    geoms = np.asarray(da[geom_dim].values, dtype=object)
    crs = _get_crs(da, geom_dim)
    columns = _column_labels(da, time_dim, time_fmt)

    n_emb = da.sizes[emb_dim]

    arrow_dtype = None
    if arrow:
        value_type = pa.from_numpy_dtype(da.dtype)
        arrow_dtype = pd.ArrowDtype(pa.list_(value_type, n_emb))

    # one delayed block list per column, each aligned on the geometry chunking
    if has_time:
        per_column = [
            da.isel({time_dim: i}).data.to_delayed().ravel()  # pure getitem
            for i in range(da.sizes[time_dim])
        ]
    else:
        per_column = [da.data.to_delayed().ravel()]

    sizes = da.chunks[0]
    parts, offset = [], 0
    for j, n in enumerate(sizes):
        idx = pd.RangeIndex(offset, offset + n)
        blocks = [col[j] for col in per_column]
        parts.append(
            delayed(_partition_to_gdf)(
                blocks, geoms[offset : offset + n], columns, crs, idx, arrow_dtype
            )
        )
        offset += n

    divisions = tuple(np.cumsum((0,) + tuple(sizes)))
    divisions = divisions[:-1] + (divisions[-1] - 1,)  # last division is inclusive
    meta = _meta(columns, crs, arrow_dtype)

    gdf = ddf.from_delayed(parts, meta=meta, divisions=divisions, verify_meta=False)

    # dispatch should already have produced a spatial frame; be explicit if not
    if not isinstance(gdf, dask_geopandas.GeoDataFrame):
        gdf = dask_geopandas.from_dask_dataframe(ddf, geometry="geometry")

    schema = pa.schema(
        [("geometry", pa.binary())]
        + [(c, pa.list_(pa.float32(), n_emb)) for c in columns]
    )
    gdf.to_parquet(path, schema=schema, write_index=False)


def _save_as_parquet(datacube: xr.DataArray, path: Path) -> bool:
    geometry_dim = dim_utils.get_geometry_dim_name(datacube)
    time_dim_name = dim_utils.get_time_dim_name(datacube)
    emb_dim_name = dim_utils.get_embedding_dim_name(datacube)

    _vector_cube_to_gdf(datacube, path, geometry_dim, emb_dim_name, time_dim_name)


def _update_stac_metadata_vector_cube(stac_metadata: dict, datacube: xr.DataArray):
    return stac_metadata


def _save_metadata_file(stac_metadata: dict, metadata_path: Path) -> bool:
    try:
        with open(metadata_path, "w") as file:
            json.dump(stac_metadata, file, indent=4)
        return True
    except Exception as e:
        raise Exception("Failed saving the metadata file.")


def save_embeddings(data: xr.DataArray) -> bool:
    # you can call this method form your project-specific save-results process
    # if this method returns True, saving was successful, you can skip your own save-result code
    # if it returns False, saving was unsuccessful (i.e. no embeddings DC) and you can run your own save-result code

    if "embedding" not in data.dims:
        raise DimensionMissing(
            "Datacube does not contain an embedding dimension. It therefore can not "
            "be used in the save_embeddings process"
        )

    _id = str(uuid4())
    result_dir = Path(OPENEO_RESULTS_PATH) / _id
    metadata_path = result_dir / "result.json"

    stac_metadata = _get_stac_item_template(_id)

    data_saved = False

    if dim_utils.is_raster_datacube(data):
        # this implies embeddings in a regular raster -> save as zarr
        zarr_out_path = result_dir / "result.zarr"
        data.name = "embeddings"
        _update_stac_metadata_raster_cube(stac_metadata, data, result_dir)
        zipped_zarr_path = _save_as_zarr(data, result_dir, zarr_out_path)
        stac_metadata = _set_stac_embedding_asset_metadata_raster(
            stac_metadata, zipped_zarr_path
        )
        data_saved = True

    if dim_utils.is_vector_datacube(data):
        # this implieds embeddings in irregular raster -> save as geo-parquet
        parquet_out_path = result_dir / "result.geoparquet"
        # _update_stac_metadata_vector_cube(stac_metadata, data)
        _save_as_parquet(data, parquet_out_path)

        data_saved = True

    if not data_saved:
        raise Exception(
            "Could not save embedding because datacube is of unknown type. Must be "
            "either a raster datacube (x and y dimensions) or a vector datacube "
            "geometry dimension"
        )

    saved = _save_metadata_file(stac_metadata, metadata_path)
    return saved
