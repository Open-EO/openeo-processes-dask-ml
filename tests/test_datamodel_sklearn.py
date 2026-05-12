from unittest.mock import MagicMock, mock_open, patch

from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR
from openeo_processes_dask_ml.process_implementations.data_model import (
    scikit_learn_model,
)


def test_rf_class_model_init_model():
    model_id = "asdf"
    model_path = f"{MODEL_CACHE_DIR}/{ model_id}.pkl"

    with patch("builtins.open", mock_open()) as mocked_open:
        rf_model_path = scikit_learn_model.RfClassModel.init_model(
            "sqrt", 100, model_id, 42
        )

        # Assert open was called once with correct args
        mocked_open.assert_called_once_with(model_path, "wb")

    assert rf_model_path == model_path
