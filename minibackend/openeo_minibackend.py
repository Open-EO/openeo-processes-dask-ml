import importlib
import inspect
import logging

from openeo_pg_parser_networkx import ProcessRegistry
from openeo_pg_parser_networkx.process_registry import Process
from openeo_processes_dask.specs import load_collection as load_collection_spec
from openeo_processes_dask.specs import save_result as save_result_spec

from minibackend.custom_processes import load_collection, save_result

# This import will yield a warning that ML capabilities are disabled
# Suppressing this warning to not confuse the user.
# Instead, ML capabilities are coming from openeo_processes_dask_ml
logging.disable(logging.CRITICAL)
from openeo_processes_dask.process_implementations.core import process

logging.disable(logging.NOTSET)  # re-enable logging

logger = logging.getLogger(__name__)

process_registry = ProcessRegistry(wrap_funcs=[process])


def register_processes(process_module: str, spec_module: str):
    # Import processes from openeo_processes_dask and register them into
    processes_from_module = [
        func
        for _, func in inspect.getmembers(
            importlib.import_module(process_module),
            inspect.isfunction,
        )
    ]

    specs_module = importlib.import_module(spec_module)
    specs = {
        func.__name__: getattr(specs_module, func.__name__)
        for func in processes_from_module
    }

    # print(specs.keys())

    for func in processes_from_module:
        process_registry[func.__name__] = Process(
            spec=specs[func.__name__], implementation=func
        )


# register the standard processes from openeo-processes-dask
register_processes(
    "openeo_processes_dask.process_implementations", "openeo_processes_dask.specs"
)

# register the new ML processes from this repo
register_processes(
    "openeo_processes_dask_ml.process_implementations", "openeo_processes_dask_ml.specs"
)

# override openeo-processes-dask implementations with my own
register_processes(
    "openeo_processes_dask_ml.process_implementations.overrides",
    "openeo_processes_dask.specs",
)


process_registry["load_collection"] = Process(
    spec=load_collection_spec, implementation=load_collection
)
process_registry["save_result"] = Process(
    spec=save_result_spec, implementation=save_result
)
