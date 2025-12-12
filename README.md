# OpenEO Processes Dask: Machine Learning

`openeo-processes-dask-ml` is a Python package that implements generic machine learning
(ML) processes for openEO. It is built to work alongside and integrate with
[openeo-processes-dask](https://github.com/Open-EO/openeo-processes-dask), extending it
by machine learning capabilities.

> [!WARNING]
> This package is Work-In-Progress, and everything is experimental.
> You will likely encounter many `NotImplementedError`.\
> At the moment, it is rather a proof-of-concept than something to be used in production.

## Currently supported
- Loading pre-trained ML models using the `load_ml_model` process
- Use the loaded ML model to make predictions from the datacube using `ml_predict`
- Restructure ML model output back into a datacube structure.

## Installation

This package is not published on PyPI yet. It can only be used from source

## Development environment

1. Clone the repository
2. Install it using [poetry](https://python-poetry.org/docs/):
   `poetry install --all-extras`
3. Run the test suite: `poetry run pytest`

### Extensibility

This package is made to be easily extensible (e.g. adding support for new
ML frameworks) by inheriting from
`openeo_processes_dask_ml.process_implementations.data_model.data_model.MLModel` and implementing
the abstract methods.

### Pre-commit hooks

This repo makes use of [pre-commit](https://pre-commit.com/) hooks to enforce linting &
a few sanity checks. In a fresh development setup, install the hooks using
`poetry run pre-commit install`. These will then automatically be checked against your
changes before making the commit.

## Structure

- `minibackend` has a minimal backend implementation for executing process graphs
- `opd-ml-dev-utils` has some scripts that are helpful during development
- `openeo-processes-dask-ml` the actuall ML process specs and implementations
- `tests` for pytest

## Acknowledgement

Development within this repository are carried out as part of the
[Embed2Scale](https://embed2scale.eu/) project and is cofunded by the EU Horizon Europe
program under Grant Agreement 101131841. Additional funding for this project has been
provided by the Swiss State Secretariat for Education, Research and Innovation and UK
Research and Innovation.
