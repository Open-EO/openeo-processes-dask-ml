import logging
from collections.abc import Iterable
from multiprocessing import Process
from pathlib import Path

import numpy as np
import torch

from openeo_processes_dask_ml.model_execution._argparser import get_parser
from openeo_processes_dask_ml.model_execution._file_chunk import get_file_chunk
from openeo_processes_dask_ml.process_implementations.utils.proc_expression_utils import (
    run_expression,
)

logger = logging.getLogger(__name__)


def load_and_preprocess(file_path: Path, preproc_expression) -> torch.Tensor:
    inp = np.load(file_path)

    if preproc_expression is not None:
        try:
            preproc_batch = run_expression(inp, preproc_expression)
            tensor = torch.from_numpy(preproc_batch)
        except:
            batch_tensor = torch.from_numpy(inp)
            tensor = run_expression(batch_tensor, preproc_expression)
    else:
        tensor = torch.from_numpy(inp)

    return tensor


def make_prediction(model_on_device, inp):
    with torch.no_grad():
        out = model_on_device(inp)
    return out


def postprocess(model_output, postproc_expression) -> torch.Tensor:
    if postproc_expression is not None:
        out_postproc = run_expression(model_output, postproc_expression)
    else:
        out_postproc = model_output

    if out_postproc.device != "cpu":
        out_postproc = out_postproc.cpu()

    return out_postproc


def predict(
    cuda_id: int,
    model_path: str,
    tmp_dir_output: Path,
    file_chunk: Iterable[Path],
    preproc_expression=None,
    postproc_expression=None,
):
    device = f"cuda:{cuda_id}"
    model = torch.jit.load(model_path).to(device).eval()

    for file_path in file_chunk:
        tensor = load_and_preprocess(file_path, preproc_expression)
        tensor = tensor.to(device)

        out = make_prediction(model, tensor)

        out_postproc = postprocess(out, postproc_expression)

        out_cube = out_postproc.numpy()

        np.save(tmp_dir_output / file_path.name, out_cube)

    return True


def start_prediction_processes(
    model_path: str,
    tmp_dir_input: Path,
    tmp_dir_output: Path,
    preproc_expression=None,
    postproc_expression=None,
    n_cuda_devices: int = None,
):
    if n_cuda_devices is None:
        # default: use all devices available
        n_cuda_devices = torch.cuda.device_count()
    else:
        devices_available = torch.cuda.device_count()
        if n_cuda_devices > devices_available:
            raise ValueError(
                f"Not enough cuda devices: Requested {devices_available} "
                f"but {n_cuda_devices} were requested."
            )
    logger.error(
        f"CUDA Devices: {n_cuda_devices}",
    )
    processes = []
    for cuda_id in range(n_cuda_devices):
        file_chunk = get_file_chunk(tmp_dir_input, cuda_id, n_cuda_devices)
        p = Process(
            target=predict,
            args=(
                cuda_id,
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
        args.n_cuda_devices,
    )
    print(ex)
