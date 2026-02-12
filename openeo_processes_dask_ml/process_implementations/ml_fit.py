import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.data_model import MLModel
from openeo_processes_dask_ml.process_implementations.utils.dim_utils import (
    band_dim_options,
)


def ml_fit(model: MLModel, training_set: xr.DataArray, target: str):
    # simplest case, just for now, not generic:

    # training_set is a vector DataCube, i.e. one geometry dimension
    # trainign_set has a "bands" dimension.
    # training_set has target as non-dimensional coordiantes for the
    # geometry-dimension

    print("fit model")

    if "geometry" not in training_set.dims:
        raise DimensionMissing("No geoemtry dimension in training_set")

    if target not in training_set.coords:
        raise DimensionMissing(f"Target {target} not in training_set")

    inp_dims = model.input.input.dim_order
    for inp_dim in inp_dims:
        if inp_dim not in training_set.dims:
            raise DimensionMissing(
                f"Dimension {inp_dim} required by the model is not in training_set"
            )

    return model
