import builtins
import os
from unittest.mock import patch

import pytest
import xarray as xr
from dask import array as da
from dask.delayed import Delayed

from openeo_processes_dask_ml.process_implementations import (
    ml_fit,
    ml_predict,
    mlm_class_random_forest,
)
from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR


@pytest.fixture
def training_data() -> xr.DataArray:
    data = xr.DataArray(
        da.random.random((999, 5, 12)),
        dims=("geometry", "time", "bands"),
        coords={
            "geometry": range(999),
            "class_name": ("geometry", [*333 * ["a"], *333 * ["b"], *333 * ["c"]]),
            "bands": [f"B{str(b)}" for b in range(12)],
        },
    )
    data.name = None
    return data


@pytest.fixture
def predict_data() -> xr.DataArray:
    dc = xr.DataArray(
        da.random.random((5, 12, 10, 10)),
        dims=("time", "band", "x", "y"),
        coords={
            "band": [f"B{str(b)}" for b in range(12)],
        },
    )
    dc.name = None
    return dc


@pytest.mark.vcr()
def test_init_model_default_values():
    seed = 42
    model = mlm_class_random_forest("log2", 100, seed)

    assert model.seed == seed
    assert model._model_filepath is not None
    assert isinstance(model._model_filepath, str)

    assert os.path.exists(model._model_filepath)

    assert model.model_metadata.pretrained == False

    assert "time" in model.input.input.dim_order
    assert "bands" in model.input.input.dim_order

    os.remove(model._model_filepath)


@pytest.mark.parametrize("dim_name", ("band", "bands", "embedding", "embeddings"))
def test_init_model_manual_values(dim_name: str):
    model = mlm_class_random_forest("log2", 50, dimension=dim_name)
    assert len(model.input.input.dim_order) == 2
    assert "time" in model.input.input.dim_order
    assert dim_name in model.input.input.dim_order

    os.remove(model._model_filepath)


def test_init_model_no_temporal():
    model = mlm_class_random_forest(
        "log2", 50, dimension="embedding", use_timeseries=False
    )

    assert len(model.input.input.dim_order) == 1
    assert "embedding" in model.input.input.dim_order
    assert "time" not in model.input.input.dim_order

    os.remove(model._model_filepath)


# Test of the entire workflow
# 1) init RF model, 2) train RF model, 3) predict
# this is by no means systematic testing, or modular or anything
# shoild add proper systematic testing later, but this is better than nothing
@pytest.mark.vcr()
def test_fit_model_default_values(
    training_data: xr.DataArray, predict_data: xr.DataArray
):
    # 1) init model
    model = mlm_class_random_forest("log2")

    untrained_filepath = model._model_filepath

    with open(untrained_filepath, "rb") as file:
        untrained_model_bytes = file.read()

    # 2) fit model
    fitted = ml_fit(model, training_data, "class_name")

    assert fitted.input.input.dim_order == ["time", "bands"]
    assert fitted.input.input.shape == [
        len(training_data.coords["time"]),
        len(training_data.coords["bands"]),
    ]

    assert fitted.input.input.shape == [5, 12]

    assert fitted.output.result.dim_order == ["class_name"]
    assert fitted.output.result.shape == [1]

    assert fitted.model_metadata.pretrained == True

    assert isinstance(fitted._model_filepath, Delayed)

    out_path = fitted._model_filepath.compute()

    assert isinstance(out_path, str)
    assert out_path == untrained_filepath

    with open(out_path, "rb") as file:
        trained_model_bytes = file.read()

    # make usre the file has changed
    assert untrained_model_bytes != trained_model_bytes

    # 3) predict
    out = ml_predict(predict_data, fitted)

    assert isinstance(out, xr.DataArray)
    assert "bands" not in out.dims
    assert "time" not in out.dims

    assert "class_name" in out.dims
    assert len(out.coords["class_name"]) == 1

    assert len(predict_data.coords["x"]) == len(out.coords["x"])
    assert len(predict_data.coords["y"]) == len(out.coords["y"])

    out = out.compute()
    assert isinstance(out, xr.DataArray)

    os.remove(out_path)


@pytest.mark.vcr()
def test_fit_model_no_temporal(training_data: xr.DataArray, predict_data: xr.DataArray):
    # 1) init model
    model = mlm_class_random_forest("log2", use_timeseries=False)

    untrained_filepath = model._model_filepath

    with open(untrained_filepath, "rb") as file:
        untrained_model_bytes = file.read()

    # 2) fit model
    fitted = ml_fit(model, training_data, "class_name")

    assert len(fitted.input.input.dim_order) == 1
    assert fitted.input.input.dim_order == ["bands"]
    assert "time" not in fitted.input.input.dim_order

    assert fitted.input.input.shape == [12]

    assert isinstance(fitted._model_filepath, Delayed)

    out_path = fitted._model_filepath.compute()

    assert isinstance(out_path, str)
    assert out_path == untrained_filepath

    with open(out_path, "rb") as file:
        trained_model_bytes = file.read()

    # make usre the file has changed
    assert untrained_model_bytes != trained_model_bytes

    # 3) predict
    out = ml_predict(predict_data, fitted)

    assert isinstance(out, xr.DataArray)
    assert "bands" not in out.dims
    assert "time" in out.dims

    assert "class_name" in out.dims
    assert len(out.coords["class_name"]) == 1
    assert len(out.coords["time"]) == 5

    assert len(predict_data.coords["x"]) == len(out.coords["x"])
    assert len(predict_data.coords["y"]) == len(out.coords["y"])

    out = out.compute()
    assert isinstance(out, xr.DataArray)

    print(out)

    os.remove(out_path)
