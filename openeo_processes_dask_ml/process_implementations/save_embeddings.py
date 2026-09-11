import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import dask
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import xarray as xr
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


# ----------------------------------------------------


def _get_crs(da: xr.DataArray, geom_dim: str):
    try:
        crs = da.xvec.crs
        return crs.get(geom_dim) if isinstance(crs, dict) else crs
    except Exception:
        return da[geom_dim].attrs.get("crs")


def _column_labels(da, time_dim, time_fmt) -> list[str]:
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


def _geometry_types(geoms) -> list[str]:
    ids = shapely.get_type_id(geoms)
    zs = shapely.has_z(geoms).astype(np.int8)
    combos = np.unique(np.stack([ids, zs], axis=1), axis=0)
    out = set()
    for tid, z in combos:
        name = _TYPE_NAMES.get(int(tid))
        if name:
            out.add(f"{name} Z" if z else name)
    return sorted(out)


def _geo_metadata(geoms, crs, column="geometry") -> dict:
    """GeoParquet 1.1 file metadata."""
    xmin, ymin, xmax, ymax = shapely.total_bounds(geoms)
    col = {
        "encoding": "WKB",
        "geometry_types": _geometry_types(geoms),
        "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "crs": CRS.from_user_input(crs).to_json_dict() if crs is not None else None,
    }
    return {"version": "1.1.0", "primary_column": column, "columns": {column: col}}


def _build_schema(columns, value_type, n_emb, geo_meta) -> pa.Schema:
    fields = [pa.field("geometry", pa.binary())]
    fields += [pa.field(c, pa.list_(value_type, n_emb)) for c in columns]
    # NOTE: no b"pandas" key -> nothing to misparse on read
    return pa.schema(fields, metadata={b"geo": json.dumps(geo_meta).encode()})


def _fsl(block: np.ndarray) -> pa.FixedSizeListArray:
    """(n, k) ndarray -> FixedSizeListArray, no copy of the values."""
    block = np.ascontiguousarray(np.asarray(block))
    n, k = block.shape
    return pa.FixedSizeListArray.from_arrays(pa.array(block.reshape(-1)), k)


def _blocks_to_table(blocks, geoms, schema) -> pa.Table:
    """Runs inside a dask task: geometry chunk + one block per time slice."""
    wkb = shapely.to_wkb(np.asarray(geoms, dtype=object), flavor="iso")
    arrays = [pa.array(wkb, type=pa.binary())] + [_fsl(b) for b in blocks]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_vector_cube_parquet(
    da: xr.DataArray,
    path: Path,
    geom_dim: str = "geometry",
    emb_dim: str = "embedding",
    time_dim: str | None = "time",
    time_fmt=None,
    compression: str = "zstd",
    compression_level=None,
    row_group_size=None,
    partitioned=False,  # True -> one file per geometry chunk (parallel)
):
    for dim in (geom_dim, emb_dim):
        if dim not in da.dims:
            raise ValueError(f"Missing required dimension {dim!r}; dims={da.dims}")
    extra = set(da.dims) - {geom_dim, emb_dim, time_dim}
    if extra:
        raise ValueError(f"Unexpected extra dimension(s): {sorted(extra)}")

    has_time = time_dim is not None and time_dim in da.dims  # <- fixed
    order = (geom_dim, time_dim, emb_dim) if has_time else (geom_dim, emb_dim)
    da = da.transpose(*order)

    if len(da.chunks[da.get_axis_num(emb_dim)]) > 1:
        da = da.chunk({emb_dim: -1})

    geoms = np.asarray(da[geom_dim].values, dtype=object)
    crs = _get_crs(da, geom_dim)
    columns = _column_labels(da, time_dim, time_fmt)

    schema = _build_schema(
        columns,
        pa.from_numpy_dtype(da.dtype),
        da.sizes[emb_dim],
        _geo_metadata(geoms, crs),
    )

    if has_time:
        per_column = [
            da.isel({time_dim: i}).data.to_delayed().ravel()
            for i in range(da.sizes[time_dim])
        ]
    else:
        per_column = [da.data.to_delayed().ravel()]

    sizes = da.chunks[0]
    offsets = np.cumsum((0,) + tuple(sizes))

    # ---- one file per chunk: fully parallel -------------------------------
    if partitioned:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        @dask.delayed
        def _write_part(blocks, part_geoms, dest):
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


def _save_as_parquet(datacube: xr.DataArray, path: Path) -> bool:
    geometry_dim = dim_utils.get_geometry_dim_name(datacube)
    emb_dim_name = dim_utils.get_embedding_dim_name(datacube)

    try:
        time_dim_name = dim_utils.get_time_dim_name(datacube)
    except DimensionMissing:
        time_dim_name = None

    try:
        write_vector_cube_parquet(
            datacube, path, geometry_dim, emb_dim_name, time_dim_name, partitioned=False
        )
        return True
    except:
        return False


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
