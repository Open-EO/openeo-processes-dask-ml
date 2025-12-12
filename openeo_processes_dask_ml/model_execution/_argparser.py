import argparse
import os
from pathlib import Path

from pystac.extensions.mlm import ProcessingExpression


def existing_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return Path(path)


def existing_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    return Path(path)


def create_expression(expression):
    return ProcessingExpression.create(format="python", expression=expression)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openeo prediction script",
        description="Predict from .npy files in a directory, saves output as .npy",
    )

    parser.add_argument(
        "torchscript_path",
        help="Path to the torchscript model which will be used for predcition",
        type=existing_file,
    )
    parser.add_argument(
        "input_dir",
        help="Input directory of .npy files form which will be predicted",
        type=existing_dir,
    )
    parser.add_argument(
        "output_dir", help="Output directory of prediction results", type=existing_dir
    )
    parser.add_argument(
        "--preprocessing_function",
        help="Python preprocessing function",
        type=create_expression,
    )
    parser.add_argument(
        "--postprocessing_function",
        help="Python postprocessing function",
        type=create_expression,
    )
    parser.add_argument(
        "--n_cuda_devices",
        help="Number of CUDA devices to use in prediction task. "
        "If left unset, all available devices will be used",
        required=False,
        type=int,
    )

    return parser
