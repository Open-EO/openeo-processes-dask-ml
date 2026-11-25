import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import dask
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import xarray as xr
import xvec  # xr xvec accessor, do not remove, even if IDE says we dont need it
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing
from pyproj import CRS

from openeo_processes_dask_ml.process_implementations.constants import (
    OPENEO_RESULTS_PATH,
)
from openeo_processes_dask_ml.process_implementations.utils import dim_utils, zip_utils

_TYPE_NAMES = {
    0: "Point",
    1: "LineString",
    2: "Polygon",
    3: "MultiPoint",
    4: "MultiLineString",
    5: "MultiPolygon",
    6: "GeometryCollection",
}

logger = logging.getLogger(__name__)


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


def _set_stac_spatial_metadata(stac_metadata: dict, bbox: list[float]):
    """
    Sets spatial metadata of the STAC Item: bbox and geometry.
    :param stac_metadata: dict of the STAC Item
    :param bbox: list of four floats, [xmin, ymin, xmax, ymax]
    """
    if len(bbox) != 4:
        raise ValueError("Length of bbox must be exactly 4: [xmin, ymin, xmax, ymax]")
    bbox_polygon = json.loads(shapely.to_geojson(shapely.box(*bbox)))
    stac_metadata["bbox"] = bbox
    stac_metadata["geometry"] = bbox_polygon


def _set_stac_spatial_metadata_raster(stac_metadata: dict, datacube: xr.DataArray):
    x_dim, y_dim = dim_utils.get_spatial_dim_names(datacube)

    # todo: convert coords to wgs84
    xmin = float(min(datacube.coords[x_dim].data))
    ymin = float(min(datacube.coords[y_dim].data))
    xmax = float(max(datacube.coords[x_dim].data))
    ymax = float(max(datacube.coords[y_dim].data))
    bbox = [xmin, ymin, xmax, ymax]
    _set_stac_spatial_metadata(stac_metadata, bbox)


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
    emb_type = datacube.attrs.get("emb:type")
    if emb_type is None:
        patch_dim_options = ["patch", "patch_x", "patch_y"]
        if any(x in patch_dim_options for x in datacube.dims):
            emb_type = "patch"

        elif dim_utils.is_vector_datacube(datacube):
            emb_type = "chip"

        elif dim_utils.is_raster_datacube(datacube):
            # TODO: distinguish between chip and pixel embs
            emb_type = "chip"

        else:
            emb_type = "UNKNOWN"
            logger.warning("Cannot reliably determine embedding type")

    stac_metadata["properties"]["emb:type"] = emb_type
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


def _set_stac_spatial_metadata_vector(stac_metadata: dict, datacube: xr.DataArray):
    geom_dim = dim_utils.get_geometry_dim_name(datacube)

    crs = datacube[geom_dim].attrs.get("crs")
    to_crs = "EPSG:4326"
    geoms = gpd.GeoSeries(datacube.coords[geom_dim].values, crs=crs).to_crs(to_crs)

    bbox = geoms.total_bounds.tolist()
    _set_stac_spatial_metadata(stac_metadata, bbox)


def _set_stac_embedding_metadata_vector(stac_metadata: dict, datacube: xr.DataArray):
    _set_stac_embedding_metadata(stac_metadata, datacube)
    stac_metadata["properties"]["emb:chip_layout"]["layout_type"] = "variable_grid"


def _update_stac_metadata_vector_cube(stac_metadata: dict, datacube: xr.DataArray):
    _set_stac_spatial_metadata_vector(stac_metadata, datacube)
    _set_stac_time_metadata(stac_metadata, datacube)
    _set_stac_embedding_metadata_vector(stac_metadata, datacube)
    return stac_metadata


def _set_stac_embedding_asset_metadata_vector(
    stac_metadata: dict, out_path: Path
) -> dict:
    stac_metadata["assets"]["embeddings"]["href"] = str(out_path.absolute())
    stac_metadata["assets"]["embeddings"]["type"] = "application/vnd.apache.parquet"
    return stac_metadata


# ----------------------------------------------------


def _get_crs(da: xr.DataArray, geom_dim: str) -> Any:
    try:
        crs = da.xvec.crs
        return crs.get(geom_dim) if isinstance(crs, dict) else crs
    except Exception:
        return da[geom_dim].attrs.get("crs")


def _column_labels(
    da: xr.DataArray, time_dim: str | None, time_fmt: str | None
) -> list[str]:
    if time_dim not in da.dims:
        return ["embedding"]
    if time_dim in da.dims and len(da.coords[time_dim].values) == 1:
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


def _geometry_types(geoms: np.ndarray) -> list[str]:
    ids = shapely.get_type_id(geoms)
    zs = shapely.has_z(geoms).astype(np.int8)
    combos = np.unique(np.stack([ids, zs], axis=1), axis=0)
    out = set()
    for tid, z in combos:
        name = _TYPE_NAMES.get(int(tid))
        if name:
            out.add(f"{name} Z" if z else name)
    return sorted(out)


def _geo_metadata(geoms: np.ndarray, crs: Any, column: str = "geometry") -> dict:
    """GeoParquet 1.1 file metadata."""
    xmin, ymin, xmax, ymax = shapely.total_bounds(geoms)
    col = {
        "encoding": "WKB",
        "geometry_types": _geometry_types(geoms),
        "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "crs": CRS.from_user_input(crs).to_json_dict() if crs is not None else None,
    }
    return {"version": "1.1.0", "primary_column": column, "columns": {column: col}}


def _build_emb_column_schema(
    embedding_shape: tuple[int, ...], value_type: pa.DataType, column_name: str
) -> pa.Field:
    dtype: pa.DataType = value_type
    for dim in reversed(embedding_shape):
        dtype = pa.list_(dtype, dim)
    return pa.field(column_name, dtype)


def _build_schema(
    columns: list[str],
    value_type: pa.DataType,
    embedding_shape: tuple[int, ...],
    geo_meta: dict,
) -> pa.Schema:
    fields = [pa.field("geometry", pa.binary())]
    fields += [
        _build_emb_column_schema(embedding_shape, value_type, c) for c in columns
    ]
    return pa.schema(fields, metadata={b"geo": json.dumps(geo_meta).encode()})


def _to_fsl(block: npt.ArrayLike) -> pa.FixedSizeListArray:
    """Convert an (n, k1, k2, ..., km) array into a nested FixedSizeListArray.

    The result has length ``n``, and each element is a fixed-size list of
    ``k1`` fixed-size lists of ``k2`` ... of ``km`` scalar values, i.e. the type is
    ``fixed_size_list(fixed_size_list(..., km), ..., k1)``.

    The values are not copied for primitive dtypes when the input is already
    C-contiguous.
    """
    contiguous_block = np.ascontiguousarray(block)

    if contiguous_block.ndim < 2:
        raise ValueError(
            f"Expected an array with at least 2 dimensions (n, k1, ...), "
            f"got shape {contiguous_block.shape}"
        )

    # Flatten to 1-D (a view, since the array is contiguous) and wrap in Arrow.
    nested_array = pa.array(contiguous_block.reshape(-1))

    # Wrap from the innermost dimension outward: km, ..., k2, k1.
    # The leading dimension n becomes the length of the outermost array.
    for list_size in reversed(contiguous_block.shape[1:]):
        nested_array = pa.FixedSizeListArray.from_arrays(nested_array, list_size)

    return nested_array


def _blocks_to_table(
    blocks: list[np.ndarray], geoms: np.ndarray, schema: pa.Schema
) -> pa.Table:
    """Runs inside a dask task: geometry chunk + one block per time slice."""
    wkb = shapely.to_wkb(np.asarray(geoms, dtype=object), flavor="iso")
    arrays = [pa.array(wkb, type=pa.binary())] + [_to_fsl(b) for b in blocks]
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_vector_cube_parquet(
    da: xr.DataArray,
    path: Path,
    geom_dim: str = "geometry",
    emb_dim: str = "embedding",
    time_dim: str | None = "time",
    time_fmt: str | None = None,
    compression: str = "zstd",
    compression_level: int | None = None,
    row_group_size: int | None = None,
    partitioned: bool = False,  # True -> one file per geometry chunk (parallel)
) -> Path | list:
    for dim in (geom_dim, emb_dim):
        if dim not in da.dims:
            raise ValueError(f"Missing required dimension {dim!r}; dims={da.dims}")

    ordered_patch_emb_dims = {"patch_x", "patch_y"}
    unordere_patch_emb_dim = "patch"

    extra = set(da.dims) - {
        geom_dim,
        emb_dim,
        time_dim,
        *ordered_patch_emb_dims,
        unordere_patch_emb_dim,
    }
    if extra:
        raise ValueError(f"Unexpected extra dimension(s): {sorted(extra)}")

    has_ordered_patch_embs = (
        True if all([p in da.dims for p in ordered_patch_emb_dims]) else False
    )
    has_unordered_patch_embs = True if unordere_patch_emb_dim in da.dims else False
    has_time = time_dim is not None and time_dim in da.dims

    if has_ordered_patch_embs and has_unordered_patch_embs:
        raise NotImplementedError(
            f"Unsupported embedding data cube: "
            f"Cube must contain either {ordered_patch_emb_dims} or "
            f"{unordere_patch_emb_dim}.\n"
            f"The given cube contains all three."
        )

    if has_ordered_patch_embs:  # has patch_x, patch_y and embedding dimensions
        emb_dims = ("patch_x", "patch_y", emb_dim)
    elif has_unordered_patch_embs:  # has patch and embedding dimensions
        emb_dims = (unordere_patch_emb_dim, emb_dim)
    else:  # only contains "embedding" dimension
        emb_dims = (emb_dim,)

    order = (time_dim, geom_dim, *emb_dims) if has_time else (geom_dim, *emb_dims)
    if da.dims != order:
        da = da.transpose(*order)

    # check if all emb_dims are in the same chunk
    is_split = any(len(da.chunks[da.get_axis_num(dim)]) > 1 for dim in emb_dims)
    if is_split:
        new_chunks = {dim: -1 for dim in emb_dims}
        da = da.chunk(new_chunks)

    geoms = np.asarray(da[geom_dim].values, dtype=object)
    crs = _get_crs(da, geom_dim)
    columns = _column_labels(da, time_dim, time_fmt)

    # shape of embedding
    # (e,) for normla embs
    # (p,e) for unordered patch embs
    # (p_x, p_y, e) for ordered patch embs
    emb_shape = tuple(da.sizes[dim] for dim in emb_dims)

    # construct pyarrow table schema
    schema = _build_schema(
        columns,
        pa.from_numpy_dtype(da.dtype),
        emb_shape,
        _geo_metadata(geoms, crs),
    )

    if has_time:
        per_column = [
            da.isel({time_dim: i}).data.to_delayed().ravel()
            for i in range(da.sizes[time_dim])
        ]
    else:
        per_column = [da.data.to_delayed().ravel()]

    geom_axis = da.get_axis_num(geom_dim)
    sizes = da.chunks[geom_axis]
    offsets = np.cumsum((0,) + tuple(sizes))

    # ---- one file per chunk: fully parallel -------------------------------
    if partitioned:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        @dask.delayed
        def _write_part(
            blocks: list[np.ndarray], part_geoms: np.ndarray, dest: str
        ) -> str:
            pq.write_table(
                _blocks_to_table(blocks, part_geoms, schema),
                dest,
                compression=compression,
                compression_level=compression_level,
            )
            return dest

        tasks = [
            _write_part(
                [col[j] for col in per_column],
                geoms[offsets[j] : offsets[j + 1]],
                str(out / f"part.{j:05d}.parquet"),
            )
            for j in range(len(sizes))
        ]
        return list(dask.compute(*tasks))

    # ---- single file: sequential, bounded memory ---------------------------
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        path,
        schema,
        compression=compression,
        compression_level=compression_level,
    ) as writer:
        for j in range(len(sizes)):
            blocks = dask.compute(*[col[j] for col in per_column])
            table = _blocks_to_table(blocks, geoms[offsets[j] : offsets[j + 1]], schema)
            writer.write_table(table, row_group_size=row_group_size)
    return path


def _save_as_parquet(datacube: xr.DataArray, path: Path):
    geometry_dim = dim_utils.get_geometry_dim_name(datacube)
    emb_dim_name = dim_utils.get_embedding_dim_name(datacube)

    try:
        time_dim_name = dim_utils.get_time_dim_name(datacube)
    except DimensionMissing:
        time_dim_name = None

    try:
        _write_vector_cube_parquet(
            datacube, path, geometry_dim, emb_dim_name, time_dim_name, partitioned=False
        )
    except Exception as e:
        raise Exception("Failed writing Geoparquet.")


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
        # this implies embeddings in irregular raster -> save as geo-parquet
        parquet_out_path = result_dir / "result.geoparquet"
        _update_stac_metadata_vector_cube(stac_metadata, data)
        _save_as_parquet(data, parquet_out_path)
        stac_metadata = _set_stac_embedding_asset_metadata_vector(
            stac_metadata, parquet_out_path
        )
        data_saved = True

    if not data_saved:
        raise Exception(
            "Could not save embedding because datacube is of unknown type. Must be "
            "either a raster datacube (x and y dimensions) or a vector datacube "
            "geometry dimension"
        )

    saved = _save_metadata_file(stac_metadata, metadata_path)
    return saved
