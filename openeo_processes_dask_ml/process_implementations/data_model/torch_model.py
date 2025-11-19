import sys
from collections.abc import Iterable
from pathlib import Path

from openeo_processes_dask_ml.model_execution import run_pytorch_model

from .data_model import MLModel


class TorchModel(MLModel):
    def make_predictions(
        self,
        model_filepath: str,
        files: Iterable[Path],
        tmp_dir_output: Path,
        preproc_expression,
        postproc_expression,
    ):
        # for running predictions directly in dask worker
        run_pytorch_model.predict(
            0,
            model_filepath,
            tmp_dir_output,
            files,
            preproc_expression,
            postproc_expression,
        )

    def get_run_command(self, tmp_dir_input, tmp_dir_output) -> list[str]:
        # command for running the python script externally
        run_command = [
            sys.executable,
            run_pytorch_model.__file__,  # script path
            self._model_filepath,
            tmp_dir_input,
            tmp_dir_output,
        ]
        return run_command
