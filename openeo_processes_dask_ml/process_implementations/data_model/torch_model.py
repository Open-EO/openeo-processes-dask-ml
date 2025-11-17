from collections.abc import Iterable
from pathlib import Path

from openeo_processes_dask_ml.model_execution.run_pytorch_model import predict

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
        predict(
            0,
            model_filepath,
            tmp_dir_output,
            files,
            preproc_expression,
            postproc_expression,
        )

    def start_subprocess_for_prediction(self):
        pass

    def submit_slurm_job_for_prediction(self):
        pass
