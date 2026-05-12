from openeo_processes_dask.process_implementations.exceptions import OpenEOException

from openeo_processes_dask_ml.process_implementations.data_model import MLModel


def save_ml_model(data: MLModel, name: str, options: dict = None):
    if data.model_metadata.pretrained is False:
        raise OpenEOException(
            "Model is not (pre)trained. Therefore we can not use it in ml_predict "
            "as the prediction would be senseless."
        )

    return data.save_model(name)
