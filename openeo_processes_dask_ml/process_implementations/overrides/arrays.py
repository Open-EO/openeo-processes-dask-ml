import xarray as xr
from openeo_processes_dask.process_implementations import (
    array_interpolate_linear as array_interpolate_linear_original,
)


def array_interpolate_linear(data: xr.DataArray):
    time_dim = data.openeo.temporal_dims[0]

    interp_data = data.interpolate_na(
        dim=time_dim, method="linear", use_coordinate=True
    )

    return interp_data
