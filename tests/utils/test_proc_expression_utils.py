import numpy as np
import pytest
import xarray as xr
from pystac.extensions.mlm import ProcessingExpression

from openeo_processes_dask_ml.process_implementations import constants
from openeo_processes_dask_ml.process_implementations.utils import proc_expression_utils

constants.ALLOW_MLM_PROCESSING_FUNCTION = True
constants.ALLOWED_MLM_PROCESSING_PACKAGES.append("tests")


@pytest.fixture
def datacube() -> xr.DataArray:
    dc = xr.DataArray(np.array((1, 2, 3)), dims=["x"], coords={"x": ["a", "b", "c"]})
    return dc


def function_for_testing(inp: xr.DataArray) -> xr.DataArray:
    # This is the function we use in the ProcessingExpression
    return inp * 2


class ClassForTesting:
    @staticmethod
    def function_in_class(datacube: xr.DataArray) -> xr.DataArray:
        return datacube * 2


@pytest.mark.parametrize(
    "expression",
    (
        "tests.utils.test_proc_expression_utils:function_for_testing",
        "tests.utils.test_proc_expression_utils:ClassForTesting.function_in_class",
    ),
)
def test_python(expression: str, datacube: xr.DataArray):
    # my_package.my_module: my_processing_function
    # my_package.my_module:MyClass.my_method

    exp_expression = expression
    proc = ProcessingExpression.create("python", exp_expression)

    new_cube = proc_expression_utils.run_process_expression(datacube, proc)

    ground_trouth = np.array((2, 4, 6))

    assert isinstance(new_cube, xr.DataArray)
    assert np.all(new_cube.data == ground_trouth)
