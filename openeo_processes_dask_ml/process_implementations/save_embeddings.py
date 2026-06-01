import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.constants import (
    OPENEO_RESULTS_PATH,
)
from openeo_processes_dask_ml.process_implementations.utils import dim_utils


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


def _save_as_parquet(datacube: xr.DataArray, path: Path) -> bool:
    pass


def save_embeddings(datacube: xr.DataArray) -> bool:
    # you can call this method form your project-specific save-results process
    # if this method returns True, saving was successful, you can skip your own save-result code
    # if it returns False, saving was unsuccessful (i.e. no embeddings DC) and you can run your own save-result code

    if "embedding" not in datacube.dims:
        raise DimensionMissing(
            "Datacube does not contain an embedding dimension. It therefore can not "
            "be used in the save_embeddings process"
        )

    _id = str(uuid4())
    out_path = Path(OPENEO_RESULTS_PATH) / _id

    spatial_dims = dim_utils.get_spatial_dim_names(datacube)
    if len(spatial_dims) == 2:
        # this implies embeddings in a regular raster -> save as zarr
        return _save_as_zarr(datacube, out_path)

    if "geometry" in datacube.dims or "geom" in datacube.dims:
        # this implieds embeddings in irregular raster -> save as geo-parquet
        return _save_as_parquet(datacube, out_path)

    return False


if __name__ == "__main__":
    from dask import array as da

    dc = xr.DataArray(
        da.random.random((2, 5, 10, 10)), dims=["time", "embedding", "y", "x"]
    )

    save_embeddings(dc)
