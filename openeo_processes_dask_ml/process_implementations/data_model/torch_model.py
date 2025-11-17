from collections.abc import Iterable
from pathlib import Path

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
        from openeo_processes_dask_ml.model_execution.run_pytorch_model import predict

        predict(
            0,
            model_filepath,
            tmp_dir_output,
            files,
            preproc_expression,
            postproc_expression,
        )

    def start_subprocess_for_prediction(
        self,
        model_filepath: str,
        tmp_dir_input: str,
        tmp_dir_output: str,
        preproc_expression,
        postproc_expression,
    ):
        import subprocess

        subproc_list = [
            "python",
            "openeo_processes_dask_ml/model_execution/run_pytorch_model.py",  # script path
            model_filepath,
            tmp_dir_input,
            tmp_dir_output,
        ]

        if preproc_expression is not None:
            subproc_list.append("--preprocessing_function")
            subproc_list.append(preproc_expression.expression)

        if postproc_expression is not None:
            subproc_list.append("--postprocessing_function")
            subproc_list.append(postproc_expression.expression)

        s = subprocess.run(subproc_list)

        if s.returncode != 0:
            raise Exception("Something went wrong in Prediction subprocess")

    def submit_slurm_job_for_prediction(self):
        pass
