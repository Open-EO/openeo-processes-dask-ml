import logging

import dask.array as da
import xarray as xr

logger = logging.getLogger(__name__)


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


def save_result(data, format="netcdf", options=None):
    # No generic implementation available, so need to implement locally!
    if format != "netcdf":
        logger.warning("Ignoring parameter 'format': Results will be saved as netcdf")
    data.attrs = {}
    data.to_netcdf("./result.nc")
    return True
