import os
from pathlib import Path


def _get_boolean_env(env_name, default_value: bool) -> bool:
    env_value = os.environ.get(env_name, None)
    if env_value is None:
        return default_value
    if env_value.lower() == "true":
        return True
    elif env_value.lower() == "false":
        return False
    else:
        raise ValueError(
            f'Env {env_name} only allows values "True" and "False". '
            f'Currently set to "{env_value}".'
        )


CACHE_DIR = os.environ.get("OPD_ML_CACHE_DIR", "./cache")
MODEL_CACHE_DIR = os.environ.get("OPD_ML_MODEL_CACHE_DIR", f"{CACHE_DIR}/model_cache")
DATACUBE_CACHE_DIR = os.environ.get(
    "OPD_ML_DATACUBE_CACHE_DIR", f"{CACHE_DIR}/datacubes"
)

TMP_DIR = os.environ.get("OPD_TMP_DIR", "./tmp")

USE_GPU = _get_boolean_env("OPD_ML_USE_GPU", True)

# OPD_MODEL_EXECUTION_MODE Env discription
#
# if mode "dask" (default): make computation directly in ONE dask worker
#   - only uses ONE GPU
#   - best for local use

# if mode "subprocess": one dask worker starts a subprocess which runs the model
#   - multi-GPU: Multi-processing in the subprocess to handle multiple GPUs
#   - Computations only in one machine -> multiple nodes in cluster not supported
#
# if mode "slurm": submit one or many slurm jobs: CURRENTLY NOT SUPPORTED
#   - n gpus per slurm job
#   - processes ends means that jobs have been submitted
#   - polling about status of the job
#   - handle dask-worker(s) timing out and restarting
#   - ...
#   - one slurm job for beginning, multiple slurm jobs later
#   - multiple slurm jobs: handle concurrency to prevent race conditions
#   - one slurm job could have finnished everything before another one started
#   - when finnished, start dask workers again and continue processing
MODEL_EXECUTION_MODE = os.environ.get("OPD_MODEL_EXECUTION_MODE", "dask")
_execution_modes = ["dask", "subprocess", "slurm"]
if MODEL_EXECUTION_MODE not in _execution_modes:
    raise ValueError(
        f"Invalid ENV OPD_MODEL_EXECUTION_MODE, must be one of "
        f"{', '.join(_execution_modes)}"
    )

SLURM_ML_CONFIG_PATH = os.environ.get("OPD_SLURM_ML_CONFIG_PATH", "./slurm_ml.config")
if MODEL_EXECUTION_MODE == "slurm" and not os.path.isfile(SLURM_ML_CONFIG_PATH):
    raise ValueError(
        f"SLURM config path OPD_SLURM_ML_CONFIG_PATH is not a valid file path: "
        f"{Path(SLURM_ML_CONFIG_PATH).resolve()}"
    )

# STAC:MLM has fields to apply a custom pre- and post-processing functions.
# - Allowing them is dangerous as it can be exploited as a remote code execution.
# - Disallowing them limits OPD-ML's versatility, as ML model outputs cannot be
#   reshaped into their final form.
ALLOW_MLM_PROCESSING_FUNCTION = _get_boolean_env(
    "OPD_ML_ALLOW_PROCESSING_FUNCTION", True
)

# Specify packages that are allowed to be used for custom pre- and post-processing.
# An empty string means everything is allowed: DANGEROUS!!!!!
_ALLOWED_MLM_PROCESSING_PACKAGES = os.environ.get(
    "OPD_ML_ALLOWED_MLM_PROCESSING_PACKAGES", "numpy;torch;ml_datacube_bridge"
)
ALLOWED_MLM_PROCESSING_PACKAGES = [
    s.strip() for s in _ALLOWED_MLM_PROCESSING_PACKAGES.split(";") if s != ""
]

S3_MODEL_REPO_ENDPOINT = os.environ.get("OPD_ML_S3_MODEL_REPO_ENDPOINT", None)
S3_MODEL_REPO_ACCESS_KEY_ID = os.environ.get("OPD_ML_S3_MODEL_REPO_ACCESS_KEY_ID", None)
S3_MODEL_REPO_SECRET_ACCESS_KEY = os.environ.get(
    "OPD_ML_S3_MODEL_REPO_SECRET_ACCESS_KEY", None
)

# make sure that both access_key_id and access_key are set
if (S3_MODEL_REPO_ACCESS_KEY_ID is None) != (S3_MODEL_REPO_SECRET_ACCESS_KEY is None):
    raise ValueError(
        "Either S3_MODEL_REPO_ACCESS_KEY_ID or S3_MODEL_REPO_SECRET_ACCESS_KEY is set "
        "not set. You must set either both, or not set any of the two."
    )
