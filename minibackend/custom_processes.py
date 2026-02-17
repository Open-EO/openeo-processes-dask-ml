import logging
import os
import uuid
import zipfile
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


RESULT_DIR = "./results/"


# # I/O processes aren't generic (yet), therefore have to custom define those.
def load_collection(
    id, spatial_extent, temporal_extent, bands=None, properties=None, **kwargs
):
    bands = [] if bands is None else bands
    properties = {} if properties is None else properties

    msg = (
        "Process 'load_collection' not implemented. Returning random numbers instead. "
        "Use process 'load_stac' for real observations instead."
    )
    logger.warning(msg)

    n_time = 10
    n_bands = 12
    n_x = 1000
    n_y = 1000

    x = xr.DataArray(
        da.random.random((n_time, n_bands, n_x, n_y)),
        dims=["time", "band", "x", "y"],
        coords={
            "time": ["t_" + str(t) for t in range(n_time)],
            "band": ["B" + str(b) for b in range(1, n_bands + 1)],
            "x": range(n_x),
            "y": range(n_y),
        },
    )
    return x


def _save_netcdf(data: xr.DataArray, filename: str) -> bool:
    data.attrs = {}
    data.to_netcdf(filename)
    return True


def _save_geotiff(data: xr.DataArray, filename: str):
    # basing this implementation on IBM's TensorLakeHouse getiff saver:
    # https://github.com/IBM/tensorlakehouse-openeo-driver/blob/main/tensorlakehouse_openeo_driver/save_result.py
    # Codee was modified to fit in here

    # Save each slice of the DataArray as a separate GeoTIFF file
    if data.openeo is not None and data.openeo.temporal_dims is not None:
        temporal_dims = data.openeo.temporal_dims
        if len(temporal_dims) > 0:
            time_dim = temporal_dims[0]
        else:
            time_dim = None
    else:
        time_dim = "time"

    if time_dim in data.dims:
        time_size = len(data[time_dim])
    else:
        time_size = 1

    driver = "COG"

    # Note: The rio.to_raster() method only works on a 2-dimensional
    # or 3-dimensional xarray.DataArray or a 2-dimensional xarray.Dataset.
    if time_size == 1:
        # destroy time dimension
        if time_dim is not None:
            data = data.isel({time_dim: 0})

        data.rio.to_raster(
            filename,
            driver=driver,  # Write driver
            reading_driver=driver,  # Read driver
        )
    else:
        path = Path(filename)
        parent_dir = path.parent
        # save as zip instead of tif
        filename = filename.replace(".gtiff", ".zip")
        # self.format = "ZIP"
        geotiff_files = list()
        time_list = list(data[time_dim].values)
        for index in range(0, time_size):
            t: np.datetime64
            t = time_list[index]
            timestamp = pd.Timestamp(t)
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
            unique_id = uuid.uuid4().hex
            output_filename = (
                parent_dir / f"{'openeo_output_'}_{timestamp_str}_{unique_id}.tif"
            )
            slice_array = data.isel({time_dim: index})
            if not np.isnan(slice_array.data).all():
                slice_array.rio.to_raster(output_filename)
                geotiff_files.append(output_filename)
        # Create a zip file and add GeoTIFF files to it
        with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for geotiff_file in geotiff_files:
                zipf.write(geotiff_file)
        # Remove the temporary GeoTIFF files
        for geotiff_file in geotiff_files:
            os.remove(geotiff_file)

    return filename


def save_result(data: xr.DataArray, format: str, options=None):
    # No generic implementation available, so need to implement locally!

    format = format.lower()

    result_id = uuid.uuid4()
    out_dir = RESULT_DIR + str(result_id) + "/"

    os.makedirs(out_dir, exist_ok=True)

    if format == "netcdf":
        filename = out_dir + "result.nc"
        saved = _save_netcdf(data, filename)

    elif format == "gtiff":
        filename = out_dir + "result.gtiff"
        saved = _save_geotiff(data, filename)

    else:
        raise NotImplementedError(f"Format {format} not supported")

    return saved
