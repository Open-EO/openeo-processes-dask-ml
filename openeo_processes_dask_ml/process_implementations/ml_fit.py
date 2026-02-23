import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing
from scipy.signal.windows import triang

from openeo_processes_dask_ml.process_implementations.data_model import MLModel


def ml_fit(model: MLModel, training_set: xr.DataArray, target: str):
    # simplest case, just for now, not generic:

    # training_set is a vector DataCube, i.e. one geometry dimension
    # trainign_set has a "bands" dimension.
    # training_set has target as non-dimensional coordiantes for the
    # geometry-dimension

    # replace geometry dimension with target dimension, drop all other properties
    cleaned = training_set.swap_dims({"geometry": target}).reset_coords(drop=True)

    if "geometry" not in training_set.dims:
        raise DimensionMissing("No geoemtry dimension in training_set")

    if target not in training_set.coords:
        raise DimensionMissing(f"Target {target} not in training_set")

    inp_dims = model.input.input.dim_order
    for inp_dim in inp_dims:
        if inp_dim not in training_set.dims:
            raise DimensionMissing(
                f"Dimension {inp_dim} required by the model is not in training_set"
            )

    # add bands metadata from datacube
    if "band" in model.input.input.dim_order or "bands" in model.input.input.dim_order:
        training_set_dims = training_set.dims

        model_band_dim_name = (
            "band" if "band" in model.input.input.dim_order else "bands"
        )
        model_band_index = model.input.input.dim_order.index(model_band_dim_name)

        if "band" not in training_set_dims and "bands" not in training_set_dims:
            raise DimensionMissing(
                "Training Dataset does not contain a bands dimension"
            )
        band_dim_name = "band" if "band" in training_set_dims else "bands"

        # set band names
        band_names = [*training_set.coords[band_dim_name].data]  # make a copy

        # set band length
        model.input.bands = band_names
        model.input.input.shape[model_band_index] = len(band_names)

    model.output.result.dim_order = [target]
    fitted_model = model.fit_model(cleaned)

    # model is not fitted, so we can adjust the pretraind parameter
    fitted_model.model_metadata.pretrained = True

    # todo: update eo:bands in asset metadata
    # todo: uddate bbox and datetime (start-end) with values from training-set

    return fitted_model
