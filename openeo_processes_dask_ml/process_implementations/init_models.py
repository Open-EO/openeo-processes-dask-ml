from datetime import datetime
from uuid import uuid4

import pystac

AVAILABLE_ML_FRAMEWORKS: list[str] = []
try:
    from openeo_processes_dask_ml.process_implementations.data_model import RfClassModel

    AVAILABLE_ML_FRAMEWORKS.append("scikit-learn")
except ModuleNotFoundError:
    pass

MODEL_NOT_CREATED_YET = "lol"


def mlm_class_random_forest(
    max_variables: int | str, num_trees: int = 100, seed: int | None = None
) -> RfClassModel:
    model_id = f"class_rf_{str(uuid4())}"

    if "scikit-learn" not in AVAILABLE_ML_FRAMEWORKS:
        raise NotImplementedError("Model class currently not supported")

    # 1) Create model object
    # map openeo max_variable param to sklearn's max_feature param
    if max_variables in ["sqrt", "log2"]:
        max_features = max_variables
    elif max_variables == "all":
        max_features = None
    elif max_variables == "onethird":
        max_features = 0.33
    else:
        raise ValueError(
            f"Unsupported value {str(max_variables)} for parameter max_variables."
        )

    # creaet sklearn RandomForest object
    model_path = RfClassModel.init_model(max_features, num_trees, model_id)

    # 2) Create stac-mlm item
    # 2a) stac-mlm props
    mlm_props = {
        "mlm:name": "RF_Classifier",
        "mlm:tasks": ["classification"],
        "mlm:architecture": "RandomForest",
        "mlm:framework": "scikit-learn",
        "mlm:batch_size_suggestion": 1,
        "mlm:pretrained": False,
        "mlm:input": [
            {
                "name": "12-Band Sentinel 2",
                "bands": [
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B11",
                    "B12",
                ],
                "input": {"shape": [12], "dim_order": ["band"], "data_type": "float16"},
                "value_scaling": [{"type": "min-max", "minimum": 0, "maximum": 1}],
                "resize_type": None,
                "pre_processing_function": None,
            }
        ],
        "mlm:output": [
            {
                "name": "rf_classification",
                "tasks": ["classification"],
                "result": {
                    "shape": [1],
                    "dim_order": ["classification"],
                    "data_type": "int16",
                },
                "post_processing_function": None,
            }
        ],
    }

    # 2b) stac-mlm model asset object
    asset = pystac.Asset(
        href=MODEL_NOT_CREATED_YET,
        title="Serialized RF Model",
        media_type="application/octet-stream; application=scikit-learn",
        roles=["mlm:model", "mlm:weights"],
        extra_fields={
            "mlm:artifact_type": "pickle.dump",
            "raster:bands": [
                {
                    "name": "B01",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 60,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B02",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 10,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B03",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 10,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B04",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 10,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B05",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 20,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B06",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 20,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B07",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 20,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B08",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 10,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B8A",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 20,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B10",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 60,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B11",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 20,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
                {
                    "name": "B12",
                    "nodata": 0,
                    "data_type": "uint16",
                    "bits_per_sample": 15,
                    "spatial_resolution": 20,
                    "scale": 0.0001,
                    "offset": 0,
                    "unit": "m",
                },
            ],
        },
    )

    # 2c) combine to stac-mlm Item
    mlm_item = pystac.Item(
        id=model_id,
        geometry={
            "type": "Polygon",
            "coordinates": [
                [[-180, -90], [-180, 90], [180, 90], [-180, 90], [-180, -90]]
            ],  # init as global, we can assign training data bbox later
        },
        bbox=[-180, -90, 180, 90],
        datetime=datetime.now(),  # to be replaced with datetime of training, use start and end to write "temporal box" later
        properties={"description": "A RF classifier model", **mlm_props},
        stac_extensions=[
            "https://stac-extensions.github.io/mlm/v1.4.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
        ],
        assets={"weights": asset},
    )

    mlm_item.validate()

    rf_model = RfClassModel(mlm_item, "weights")

    return rf_model
