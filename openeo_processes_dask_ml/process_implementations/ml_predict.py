import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import OpenEOException

from .data_model import MLModel


def ml_predict(data: xr.DataArray, model: MLModel) -> xr.DataArray:
    if model.model_metadata.pretrained is False:
        raise OpenEOException(
            "Model is not (pre)trained. Therefore we can not use it in ml_predict "
            "as the prediction would be senseless."
        )

    out = model.run_model(data)
    return out
