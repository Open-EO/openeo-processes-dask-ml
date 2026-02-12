import xarray as xr
from openeo_processes_dask.process_implementations.cubes.indices import (
    ndvi as ndvi_original,
)
from openeo_processes_dask.process_implementations.data_model import RasterCube
from openeo_processes_dask.process_implementations.exceptions import BandExists


def ndvi(data: RasterCube, nir="nir", red="red", target_band=None):
    # There is a bug in openeo-processes-dask's ndvi implementation
    # It crashes when we set the target_band
    # https://github.com/Open-EO/openeo-processes-dask/issues/362

    # Here we provide an implementation for as long as this issue remains open

    # We therefore wrap the original implementation with target_band=None, then
    # implement the behavior on our own

    nd = ndvi_original(data, nir, red, target_band=None)

    band_dim = data.openeo.band_dims[0]

    if target_band is not None:
        if target_band in data.coords:
            raise BandExists("A band with the specified target name exists.")
        nd = nd.expand_dims(band_dim).assign_coords({band_dim: [target_band]})

        # THIS IS WHERE THE BUG IS IN THE ORIGINAL
        nd = xr.concat([data, nd], dim=band_dim)

    return nd
