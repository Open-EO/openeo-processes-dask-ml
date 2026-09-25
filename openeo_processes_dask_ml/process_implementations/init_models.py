from datetime import datetime
from uuid import uuid4

import pystac

AVAILABLE_ML_FRAMEWORKS: list[str] = []
try:
    from openeo_processes_dask_ml.process_implementations.data_model import (
        RfClassModel,
        RfRegrModel,
    )

    AVAILABLE_ML_FRAMEWORKS.append("scikit-learn")
except ImportError:
    pass

MODEL_NOT_CREATED_YET = "lol"


def _resolve_max_variables(max_variables: str):
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
    return max_features


def _get_scikit_learn_mlm_props(
    dimension: str, use_timeseries: bool, mlm_name: str, task_name: str
) -> dict:
    input_dimensions = [dimension]
    input_shape = [1]
    if use_timeseries:
        input_dimensions.insert(0, "time")
        input_shape.append(1)

    output_name = f"{task_name}"

    mlm_props = {
        "mlm:name": mlm_name,
        "mlm:tasks": [task_name],
        "mlm:architecture": "RandomForest",
        "mlm:framework": "scikit-learn",
        "mlm:pretrained": False,
        "mlm:input": [
            {
                "name": "12-Band Sentinel 2",
                "bands": [],  # Fill later from Training data (if applicable)
                "input": {
                    "shape": input_shape,  # will be overwritten later
                    "dim_order": input_dimensions,
                    "data_type": "float16",
                },
                "value_scaling": None,
                "resize_type": None,
                "pre_processing_function": None,
            }
        ],
        "mlm:output": [
            {
                "name": output_name,
                "tasks": [task_name],
                "result": {
                    "shape": [1],
                    "dim_order": ["xxx"],  # will be overwritten later
                    "data_type": "int8",
                },
                "post_processing_function": None,
            }
        ],
    }
    return mlm_props


def _get_scikit_learn_mlm_asset(dimension: str) -> pystac.Asset:
    asset = pystac.Asset(
        href=MODEL_NOT_CREATED_YET,
        title="Serialized RF Model",
        media_type="application/octet-stream; application=scikit-learn",
        roles=["mlm:model", "mlm:weights"],
        extra_fields={"mlm:artifact_type": "pickle.dump"},
    )

    if dimension == "bands":
        bands_extra_fields = [
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
        ]
        asset.extra_fields["raster:bands"] = bands_extra_fields
    return asset


def _get_scikit_learn_mlm_item(
    model_id: str, task: str, mlm_props: dict, asset: pystac.Asset
) -> pystac.Item:
    mlm_item = pystac.Item(
        id=model_id,
        geometry={
            "type": "Polygon",
            "coordinates": [
                [[-180, -90], [-180, 90], [180, 90], [-180, 90], [-180, -90]]
            ],  # init as global, we can assign training data bbox later
        },
        bbox=[-180, -90, 180, 90],
        datetime=datetime.now(),
        # to be replaced with datetime of training, use start and end to write "temporal box" later
        properties={"description": f"A scikit-learn RF {task} model", **mlm_props},
        stac_extensions=[
            "https://stac-extensions.github.io/mlm/v1.4.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
        ],
        assets={"weights": asset},
    )
    return mlm_item


def _init_scikit_learn_tree_model(
    model_id: str, dimension: str, use_timeseries: bool, task: str, mlm_name: str
):
    # 2) Create stac-mlm item
    # 2a) stac-mlm props
    mlm_props = _get_scikit_learn_mlm_props(dimension, use_timeseries, mlm_name, task)

    # 2b) stac-mlm model asset object
    asset = _get_scikit_learn_mlm_asset(dimension)

    # 2c) combine to stac-mlm Item
    mlm_item = _get_scikit_learn_mlm_item(model_id, task, mlm_props, asset)

    mlm_item.validate()

    return mlm_item


def mlm_class_random_forest(
    max_variables: int | str,
    num_trees: int = 100,
    seed: int | None = None,
    dimension: str = "bands",
    use_timeseries: bool = True,
) -> "RfClassModel":
    model_id = f"class_rf_{str(uuid4())}"
    task = "classification"
    mlm_name = "RF_Classification"

    if "scikit-learn" not in AVAILABLE_ML_FRAMEWORKS:
        raise NotImplementedError("Model class currently not supported")

    max_features = _resolve_max_variables(max_variables)
    model_path = RfClassModel.init_model(max_features, num_trees, model_id, seed)

    mlm_item = _init_scikit_learn_tree_model(
        model_id, dimension, use_timeseries, task, mlm_name
    )

    rf_model = RfClassModel(mlm_item, "weights", 0, 0)
    rf_model.set_model_filepath(model_path)
    rf_model.seed = seed

    return rf_model


def mlm_regr_random_forest(
    max_variables: int | str,
    num_trees: int = 100,
    seed: int | None = None,
    dimension: str = "bands",
    use_timeseries: bool = True,
) -> "RfRegrModel":
    model_id = f"regr_rf_{str(uuid4())}"
    task = "regression"
    mlm_name = "RF_Regression"

    if "scikit-learn" not in AVAILABLE_ML_FRAMEWORKS:
        raise NotImplementedError("Model class currently not supported")

    max_features = _resolve_max_variables(max_variables)
    model_path = RfRegrModel.init_model(max_features, num_trees, model_id, seed)

    mlm_item = _init_scikit_learn_tree_model(
        model_id, dimension, use_timeseries, task, mlm_name
    )

    rf_model = RfRegrModel(mlm_item, "weights", 0, 0)
    rf_model.set_model_filepath(model_path)
    rf_model.seed = seed

    return rf_model
