import pickle
from collections.abc import Iterable
from multiprocessing import Process
from pathlib import Path

import numpy as np

from openeo_processes_dask_ml.model_execution._argparser import get_parser
from openeo_processes_dask_ml.model_execution._file_chunk import get_file_chunk
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


def start_prediction_processes(
    model_path: str,
    tmp_dir_input: Path,
    tmp_dir_output: Path,
    preproc_expression=None,
    postproc_expression=None,
    n_processes: int = 2,
) -> bool:
    processes = []
    for proc_id in range(n_processes):
        file_chunk = get_file_chunk(tmp_dir_input, proc_id, n_processes)
        p = Process(
            target=predict,
            args=(
                model_path,
                tmp_dir_output,
                file_chunk,
                preproc_expression,
                postproc_expression,
            ),
        )
        p.start()
        processes.append(p)

        for p in processes:
            p.join()

    return True


if __name__ == "__main__":
    # CWD must be project root for imports and everything to work

    args = get_parser().parse_args()

    ex = start_prediction_processes(
        args.model_filepath,
        args.input_dir,
        args.output_dir,
        args.preprocessing_function,
        args.postprocessing_function,
    )
