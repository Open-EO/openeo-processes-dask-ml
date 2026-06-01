import re

import pystac
from stac_validator.validate import StacValidate

AVAILABLE_ML_FRAMEWORKS: list[str] = []

from .data_model import MLModel

try:
    from .data_model import TorchModel

    AVAILABLE_ML_FRAMEWORKS.append("PyTorch")
except ImportError:
    pass

from openeo_processes_dask_ml.process_implementations.utils import stac_utils


def load_stac_ml(
    uri: str, model_asset: str = None, input_index: int = 0, output_index: int = 0
) -> MLModel:
    # there was a bug in an older version of openeo-pg-parser-networkx where integer
    # arguments were passed as floats
    # (https://github.com/Open-EO/openeo-pg-parser-networkx/issues/110)
    # Therefore, to maintain compatibility with older versions
    if isinstance(input_index, float):
        input_index = int(input_index)
    if isinstance(output_index, float):
        output_index = int(output_index)

    stac = stac_utils.load_stac_json(uri)

    # check if downloaded JSON is valid STAC
    stac_validator = StacValidate()
    stac_valid = stac_validator.validate_dict(stac)
    if not stac_valid:
        raise Exception("The provided URI does not point to a valid STAC-Item")

    # check if downloaded JSON is valid STAC Item
    stac_type = stac["type"]
    if stac_type != "Feature":
        raise Exception("The provided URI does not point to a STAC-Item.")

    # Check if downloaded STAC Item implements the STAC:MLM extension
    extensions = stac["stac_extensions"]
    regex = (
        r"^https:\/\/stac-extensions\.github\.io\/mlm\/v(\d+\.){0,2}\d*\/schema\.json$"
    )
    pattern = re.compile(regex)
    follows_mlm = any(pattern.match(s) for s in extensions)
    if not follows_mlm:
        raise Exception(
            "The provided STAC Item does not implement the STAC:MLM standard"
        )

    mlm_item = pystac.Item.from_dict(stac)

    # check if the item's extensions are valid (e
    try:
        mlm_item.validate()
    except pystac.errors.STACValidationError:
        raise Exception(
            "The provided STAC Item does not implement STAC:MLM extension correctly."
        )

    # todo: mlm:framework could be in asset

    # Check if model runtime is supported (ONNX!, torch? tf?)
    ml_framework = mlm_item.ext.mlm.framework

    if ml_framework.lower() not in [f.lower() for f in AVAILABLE_ML_FRAMEWORKS]:
        raise Exception(
            f"The ML framework {ml_framework} as required by the provided STAC:MLM Item"
            f"is not supported by this backend. Supported backends: "
            f"{', '.join(AVAILABLE_ML_FRAMEWORKS)}"
        )

    # check if input_index and output_index are valid
    if input_index >= len(mlm_item.ext.mlm.input):
        raise Exception(
            f"{input_index=} is invalid, as it exceeds the length of available input "
            f"specifications in the provided STAC:MLM item. Remember that indexes start"
            f"at 0."
        )
    if output_index >= len(mlm_item.ext.mlm.output):
        raise Exception(
            f"{output_index=} is invalid, as it exceeds the length of available output "
            f"specifications in the provided STAC:MLM item. Remember that indexes start"
            f"at 0."
        )

    if ml_framework.lower() == "pytorch":
        model_object = TorchModel(mlm_item, model_asset, input_index, output_index)
    else:
        raise Exception(f"{ml_framework} runtime is not supported.")

    return model_object
