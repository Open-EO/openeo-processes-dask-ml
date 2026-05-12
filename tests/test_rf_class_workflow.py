import os

import numpy as np
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

TARGET_DIM = "target_dim"


@pytest.fixture
def training_data() -> xr.DataArray:
    data = xr.DataArray(
        da.random.random((999, 5, 12)),
        dims=("geometry", "time", TARGET_DIM),
        coords={
            "geometry": range(999),
            "class_name": ("geometry", [*333 * ["a"], *333 * ["b"], *333 * ["c"]]),
            TARGET_DIM: [f"B{str(b)}" for b in range(12)],
        },
    )
    data.name = None
    return data


@pytest.fixture
def predict_data() -> xr.DataArray:
    dc = xr.DataArray(
        da.random.random((5, 12, 10, 10)),
        dims=("time", TARGET_DIM, "x", "y"),
        coords={
            TARGET_DIM: [f"B{str(b)}" for b in range(12)],
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
@pytest.mark.parametrize("dim_to_classify", (None, "band", "embedding"))
def test_fit_model_default_values(
    training_data: xr.DataArray, predict_data: xr.DataArray, dim_to_classify: str
):
    # 1) init model
    if dim_to_classify is None:
        model = mlm_class_random_forest("log2")
        dim_to_classify = "bands"
    else:
        model = mlm_class_random_forest("log2", dimension=dim_to_classify)

    untrained_filepath = model._model_filepath

    # rename dimensions
    training_data = training_data.rename({TARGET_DIM: dim_to_classify})
    predict_data = predict_data.rename({TARGET_DIM: dim_to_classify})

    with open(untrained_filepath, "rb") as file:
        untrained_model_bytes = file.read()

    # 2) fit model
    fitted = ml_fit(model, training_data, "class_name")

    assert len(fitted.input.input.dim_order) == 2
    assert fitted.input.input.dim_order == ["time", dim_to_classify]
    assert fitted.input.input.shape == [
        len(training_data.coords["time"]),
        len(training_data.coords[dim_to_classify]),
    ]

    assert fitted.input.input.shape == [5, 12]

    assert fitted.output.result.dim_order == ["class_name"]
    assert fitted.output.result.shape == [1]

    assert fitted.model_metadata.pretrained == True

    if dim_to_classify in ["band", "bands"]:
        assert len(fitted.input.bands) == len(training_data.coords[dim_to_classify])
        np.testing.assert_array_equal(
            fitted.input.bands, training_data.coords[dim_to_classify].values
        )
    else:
        assert len(fitted.input.bands) == 0

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
    assert dim_to_classify not in out.dims
    assert "time" not in out.dims

    assert "class_name" in out.dims
    assert len(out.coords["class_name"]) == 1

    np.testing.assert_array_equal(
        predict_data.coords["x"].values, out.coords["x"].values
    )
    np.testing.assert_array_equal(
        predict_data.coords["y"].values, out.coords["y"].values
    )

    out = out.compute()
    assert isinstance(out, xr.DataArray)

    os.remove(out_path)


@pytest.mark.vcr()
@pytest.mark.parametrize("dim_to_classify", (None, "band", "embedding"))
def test_fit_model_no_temporal(
    training_data: xr.DataArray, predict_data: xr.DataArray, dim_to_classify: str
):
    # 1) init model
    if dim_to_classify is None:
        model = mlm_class_random_forest("log2", use_timeseries=False)
        dim_to_classify = "bands"
    else:
        model = mlm_class_random_forest(
            "log2", dimension=dim_to_classify, use_timeseries=False
        )

    untrained_filepath = model._model_filepath

    # rename dimensions
    training_data = training_data.rename({TARGET_DIM: dim_to_classify})
    predict_data = predict_data.rename({TARGET_DIM: dim_to_classify})

    with open(untrained_filepath, "rb") as file:
        untrained_model_bytes = file.read()

    # 2) fit model
    fitted = ml_fit(model, training_data, "class_name")

    assert len(fitted.input.input.dim_order) == 1
    assert fitted.input.input.dim_order == [dim_to_classify]
    assert "time" not in fitted.input.input.dim_order

    assert fitted.input.input.shape == [12]

    if dim_to_classify in ["band", "bands"]:
        assert len(fitted.input.bands) == len(training_data.coords[dim_to_classify])
        np.testing.assert_array_equal(
            fitted.input.bands, training_data.coords[dim_to_classify].values
        )
    else:
        assert len(fitted.input.bands) == 0

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
    assert dim_to_classify not in out.dims
    assert "time" in out.dims

    assert "class_name" in out.dims
    assert len(out.coords["class_name"]) == 1
    assert len(out.coords["time"]) == 5

    np.testing.assert_array_equal(
        predict_data.coords["x"].values, out.coords["x"].values
    )
    np.testing.assert_array_equal(
        predict_data.coords["y"].values, out.coords["y"].values
    )
    np.testing.assert_array_equal(
        predict_data.coords["time"].values, out.coords["time"].values
    )

    out = out.compute()
    assert isinstance(out, xr.DataArray)

    os.remove(out_path)
