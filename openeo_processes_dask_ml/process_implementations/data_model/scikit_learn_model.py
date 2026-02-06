from collections.abc import Iterable
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

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
    def init_model(max_features, n_trees) -> RandomForestClassifier:
        r = RandomForestClassifier(n_trees, max_features=max_features)
        return r
