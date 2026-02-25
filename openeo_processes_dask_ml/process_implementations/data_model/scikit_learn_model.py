import copy
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Self

import numpy as np
import xarray as xr
from dask import array as da
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
        run_command = [
            sys.executable,
            run_sklearn_model.__file__,
            self._model_filepath,
            tmp_dir_input,
            tmp_dir_output,
        ]
        return run_command

    def predict_func(self, arr, model_path: str):
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # arr shape: (..., bands)
        orig_shape = arr.shape[:-1]
        n_features = arr.shape[-1]

        arr2d = arr.reshape(-1, n_features)

        pred = model.predict(arr2d)

        return pred.reshape(orig_shape)

    def run_model(self, datacube: xr.DataArray) -> xr.DataArray:
        # !!!!
        # At the moment only works for models that take in a single dim (e.g. bands)
        if len(self.input.input.dim_order) > 1:
            raise NotImplementedError(
                "this model is not supported as it takes more than one dim as input"
            )

        if len(self.output.result.dim_order) > 1:
            raise NotImplementedError(
                "this model is not supported as it outputs more than 1 dimension"
            )

        if self.output.result.shape[0] != 1:
            raise NotImplementedError(
                f"this model is not supported a its output length is "
                f"{self.output.result.shape[0]} "
                f" but only length 1 is supported."
            )

        # first check if all dims required by model are in data cube
        self.check_datacube_dimensions(datacube, ignore_batch_dim=True)

        if self._model_filepath is None:
            self._model_filepath = self._get_model()

        # resolve delayed model_filepath: Turn delayed string into 0-d dask array
        model_path_da = da.from_delayed(self._model_filepath, shape=(), dtype=object)

        input_dim_mapping = self.get_datacube_dimension_mapping(datacube)

        pre_datacube = self.preprocess_datacube(datacube)

        # dimensions: [*dims-in-model, *dims-not-in-model]
        if input_dim_mapping[0][1] != 0:
            pre_datacube = self.reorder_dc_dims_for_model_input(pre_datacube)

        pre_datacube_chunked = pre_datacube.chunk({input_dim_mapping[0][0]: -1})

        result = xr.apply_ufunc(
            self.predict_func,
            pre_datacube_chunked,
            model_path_da,
            input_core_dims=[[input_dim_mapping[0][0]], []],
            output_core_dims=[[]],
            # kwargs={"model_path": self._model_filepath},
            vectorize=False,
            dask="parallelized",
            output_dtypes=[self.output.result.data_type],
        )

        result = result.expand_dims(self.output.result.dim_order[0])

        return result


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
