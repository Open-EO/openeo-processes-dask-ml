import os
import re

from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR


def url_to_dir_string(s: str, preserve_file_extension: bool = False) -> str:
    """
    Transform a string and sanitizes it to be used as a directory name
    :param s: The string to be sanitized
    :return: The sanitized string
    """
    if preserve_file_extension:
        prefix = s.split(".")[-1]
        s = ".".join(s.split(".")[:-1])
    else:
        prefix = ""

    if s == "":
        s = "model"

    sanitized_name = re.sub(r'[\\/:*?"<>|$&;,=#.\s]', "_", s)

    # Remove leading/trailing spaces and dots introduced by replacement if any
    sanitized_name = sanitized_name.strip(" _.")

    # Check if the name (case-insensitive) matches a reserved name
    # Prepend with underscore to avoid conflict
    reserved_names = [
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
    ]
    if sanitized_name.lower() in reserved_names:
        sanitized_name = "_" + sanitized_name

    # Ensure the name is not empty after sanitization and provide default
    if not sanitized_name:
        sanitized_name = "sanitized_dir"

    # You might also want to truncate very long names if needed,
    # although modern OSes support quite long paths.

    if prefix:
        sanitized_name = sanitized_name + "." + prefix

    return sanitized_name
