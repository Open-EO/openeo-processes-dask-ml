import pickle
from collections.abc import Iterable
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

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
        pass

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
