import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import sklearn

from openeo_processes_dask_ml.process_implementations.utils.proc_expression_utils import (
    run_expression,
)


def load_and_preprocess(file_path: Path, preproc_expression) -> np.ndarray:
    inp = np.load(file_path)
    if preproc_expression is not None:
        inp = run_expression(inp, preproc_expression)

    return inp.reshape(1, -1)


def postprocess(model_output, postproc_expression) -> np.ndarray:
    if postproc_expression is not None:
        model_output = run_expression(model_output, postproc_expression)

    return model_output


def predict(
    model_filepath: str,
    tmp_dir_output: Path,
    files: Iterable[Path],
    preproc_expression=None,
    postproc_expression=None,
):
    with open(model_filepath, "rb") as file:
        model = pickle.load(file)

    for file_path in files:
        arr = load_and_preprocess(file_path, preproc_expression)
        out = model.predict(arr)
        out_postproc = postprocess(out, postproc_expression)
        np.save(tmp_dir_output / file_path.name, out_postproc)

    return True
