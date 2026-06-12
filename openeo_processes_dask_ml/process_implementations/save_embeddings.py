import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.constants import (
    OPENEO_RESULTS_PATH,
)
from openeo_processes_dask_ml.process_implementations.utils import dim_utils


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
            "gsd": None,
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


def _zip_results(source_dir: Path, zip_name: str = "result.zip") -> Path:
    """
    Archives all contents of `source_dir` into a zip file stored inside
    `source_dir`, then deletes everything except the created zip file.

    Args:
        source_dir: Path to the directory to archive.
        zip_name: Name of the resulting zip archive.

    Returns:
        Path to the created zip file.
    """
    source_path = source_dir.resolve()

    if not source_path.is_dir():
        raise ValueError(f"{source_dir} is not a valid directory")

    zip_path = source_path / zip_name

    # Create zip archive
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in source_path.rglob("*"):
            # Skip the zip file itself if it already exists
            if item == zip_path:
                continue

            # Store paths relative to source_dir
            arcname = item.relative_to(source_path)
            zf.write(item, arcname)

    # Delete everything except the zip archive
    for item in source_path.iterdir():
        if item == zip_path:
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    return zip_path


def _save_as_zarr(datacube: xr.DataArray, result_path: Path) -> bool:
    datacube.to_zarr(result_path, mode="w")
    _zip_results(result_path)
    return True


def _set_stac_spatial_metadata_raster(stac_metadata: dict, datacube: xr.DataArray):
    x_dim, y_dim = dim_utils.get_spatial_dim_names(datacube)

    # todo: convert coords to wgs84
    xmin = min(datacube.coords[x_dim].data)
    ymin = min(datacube.coords[y_dim].data)
    xmax = max(datacube.coords[x_dim].data)
    ymax = max(datacube.coords[y_dim].data)
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


def _update_stac_metadata_raster_cube(stac_metadata: dict, datacube: xr.DataArray):
    _set_stac_spatial_metadata_raster(stac_metadata, datacube)
    _set_stac_time_metadata(stac_metadata, datacube)
    _set_stac_embedding_metadata_raster(stac_metadata, datacube)


def _save_as_parquet(datacube: xr.DataArray, path: Path) -> bool:
    raise NotImplementedError("Saving of irregular embedding grids not implemented")


def _update_stac_metadata_vector_cube(stac_metadata: dict, datacube: xr.DataArray):
    pass


def save_embeddings(datacube: xr.DataArray) -> bool:
    # you can call this method form your project-specific save-results process
    # if this method returns True, saving was successful, you can skip your own save-result code
    # if it returns False, saving was unsuccessful (i.e. no embeddings DC) and you can run your own save-result code

    saved = False

    if "embedding" not in datacube.dims:
        raise DimensionMissing(
            "Datacube does not contain an embedding dimension. It therefore can not "
            "be used in the save_embeddings process"
        )

    _id = str(uuid4())
    out_path = Path(OPENEO_RESULTS_PATH) / _id
    metadata_path = out_path / f"{_id}.json"

    stac_metadata = _get_stac_item_template(_id)

    spatial_dims = dim_utils.get_spatial_dim_names(datacube)
    if len(spatial_dims) == 2:
        # this implies embeddings in a regular raster -> save as zarr
        _update_stac_metadata_raster_cube(stac_metadata, datacube)
        _save_as_zarr(datacube, out_path)
        saved = True

    if "geometry" in datacube.dims or "geom" in datacube.dims:
        # this implieds embeddings in irregular raster -> save as geo-parquet
        _update_stac_metadata_vector_cube(stac_metadata, datacube)
        _save_as_parquet(datacube, out_path)
        saved = True

    if saved:
        with open(metadata_path, "w") as file:
            json.dump(stac_metadata, file, indent=4)
        return True
    else:
        raise Exception


# todo: test this
