import argparse
import os
from collections.abc import Iterable
from glob import glob
from multiprocessing import Process
from pathlib import Path

import numpy as np
import torch

from openeo_processes_dask_ml.process_implementations.utils.proc_expression_utils import (
    run_expression,
)

n_cuda_devices = torch.cuda.device_count()


def get_file_chunk(tmp_dir_input, start_chunk: int, num_chunks: int):
    all_files = sorted(glob(os.path.join(tmp_dir_input, "*.npy")))
    file_chunk = all_files[start_chunk::num_chunks]
    return file_chunk


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
    tmp_dir_input: str,
    tmp_dir_output: str,
    preproc_expression=None,
    postproc_expression=None,
):
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


# def existing_file(path):
#     if not os.path.isfile(path):
#         raise argparse.ArgumentTypeError(
#             f"{path} is not a valid file."
#         )
#     return path
#
# def existing_dir(path):
#     if not os.path.isdir(path):
#         raise argparse.ArgumentTypeError(
#             f"{path} is not a valid directory")
#     return path
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog="ProgramName",
#         description="What the program does",
#     )
#
#     parser.add_argument(
#         "torchscript_path",
#         help="Path to the torchscript model which will be used for predcition",
#         type=existing_file
#     )
#     parser.add_argument(
#         "input_dir",
#         help="Input directory of .npy files form which will be predicted",
#         type=existing_dir
#     )
#     parser.add_argument(
#         "output_dir",
#         help="Output directory of prediction results",
#         type=existing_dir
#     )
#
#     parser.add_argument(
#         "--preprocessing_function",
#         help="Python preprocessing function",
#         required=False,
#         type=str
#     )
#
#     parser.add_argument(
#         "--postprocessing_function",
#         help="Python postprocessing function",
#         required=False,
#         type=str
#     )
#
#     args = parser.parse_args()
#     print(args)
