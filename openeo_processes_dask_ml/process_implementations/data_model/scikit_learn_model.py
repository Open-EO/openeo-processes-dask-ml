import copy
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Self

import xarray as xr
from dask import dataframe as ddf
from dask import delayed
from pystac.extensions.classification import Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from openeo_processes_dask_ml.model_execution import run_sklearn_model
from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR

from .data_model import MLModel


class SkLearnModel(MLModel):
    def make_predictions(
        self,
        model_filepath: str,
        files: Iterable[Path],
        tmp_dir_output: Path,
        preproc_expression,
        postproc_expression,
    ):
        run_sklearn_model.predict(
            model_filepath,
            tmp_dir_output,
            files,
            preproc_expression,
            postproc_expression,
        )

    def get_run_command(self, tmp_dir_input, tmp_dir_output) -> list[str]:
        pass


class RfClassModel(SkLearnModel):
    @staticmethod
    def init_model(
        max_features: int | str | float | None, n_trees: int, model_id: str
    ) -> str:
        r = RandomForestClassifier(n_trees, max_features=max_features)

        # save model to disk
        modelpath = MODEL_CACHE_DIR + "/" + model_id + ".pkl"
        with open(modelpath, "wb") as file:
            pickle.dump(r, file)

        return modelpath

    @delayed
    def fit(self, training_set_df: ddf.DataFrame) -> str:
        model_path = self._model_filepath

        with open(model_path, "rb") as file:
            model: RandomForestClassifier = pickle.load(file)

        X_train = training_set_df[self.input.bands]
        y_train = training_set_df[self.output.result.dim_order[0]]

        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)

        model.fit(X_train.values, y_train_enc)

        self.output.classes = [
            Classification.create(i, name=name)
            for i, name in enumerate(encoder.classes_)
        ]

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        return model_path

    def fit_model(self, training_set: xr.DataArray) -> Self:
        in_dims = self.input.input.dim_order
        if len(in_dims) > 1:
            raise ValueError("Only one input dimension is allowed in RF classifier")
        in_dim = in_dims[0]

        out_dims = self.output.result.dim_order
        if len(out_dims) > 1:
            raise ValueError("Only one output dimension is allowed in RF classifier")

        training_set_ds = training_set.to_dataset(dim=in_dim)
        training_set_df = training_set_ds.to_dask_dataframe()

        fitted_model_path = self.fit(training_set_df)

        rf_model_copy = copy.deepcopy(self)
        rf_model_copy._model_filepath = fitted_model_path

        return rf_model_copy
