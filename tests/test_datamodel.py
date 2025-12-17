import subprocess
import unittest.mock
import uuid
from pathlib import Path

import dask.array as da
import numpy as np
import pystac
import pytest
import xarray as xr
from dask.delayed import Delayed
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMismatch,
    DimensionMissing,
)
from pystac.extensions import mlm

from openeo_processes_dask_ml.process_implementations.exceptions import (
    LabelDoesNotExist,
)
from tests.dummy.dummy_ml_model import DummyMLModel


def test_correct_asset_selection(
    blank_stac_item, random_asset, mlm_model_asset
) -> None:
    with pytest.raises(Exception):
        d = DummyMLModel(blank_stac_item)

    blank_stac_item.add_asset("asset1", random_asset)
    with pytest.raises(Exception):
        d._get_model_asset()
    with pytest.raises(Exception):
        d._get_model_asset("asset1")

    blank_stac_item.add_asset("asset2", mlm_model_asset)
    d = DummyMLModel(blank_stac_item)
    assert d._get_model_asset().title == "model"
    assert d._get_model_asset("asset2").title == "model"

    blank_stac_item.add_asset("asset3", mlm_model_asset)
    with pytest.raises(Exception):
        d = DummyMLModel(blank_stac_item)
        d._get_model_asset()
    assert d._get_model_asset("asset3").title == "model"


@pytest.mark.vcr()
def test_get_model(mlm_item: pystac.Item, monkeypatch):
    mock_opener: unittest.mock.MagicMock = unittest.mock.mock_open()

    monkeypatch.setattr("builtins.open", mock_opener)
    monkeypatch.setattr("os.makedirs", lambda x: None)

    d = DummyMLModel(mlm_item)
    model_file_path = d._get_model()

    # assert that the method was called once
    mock_opener.assert_called_once()

    # mock path exists to use
    monkeypatch.setattr("os.path.exists", lambda x: True)

    # should not download the mdoel again as it is cached
    model_file_path = d._get_model()

    # assert that the method was STILL called only once (cached file exists)
    mock_opener.assert_called_once()


@pytest.mark.parametrize(
    "model_dim_names, dc_dim_names, idx",
    (
        (("bands", "x", "y", "time"), ("band", "x", "y", "time"), (0, 1, 2, 3)),
        (("band", "x", "y", "time"), ("band", "lon", "lat", "t"), (0, 1, 2, 3)),
        (("t", "x", "y", "channel"), ("band", "x", "y", "time"), (3, 1, 2, 0)),
        (("x", "y", "asdf"), ("x", "y", "bands", "t"), (0, 1, None)),
    ),
)
def test_get_datacube_dimension_mapping(
    mlm_item: pystac.Item,
    model_dim_names: tuple[str],
    dc_dim_names: tuple[str],
    idx: list[int | None],
):
    d = DummyMLModel(mlm_item)
    mlm_item.ext.mlm.input[0].input.dim_order = model_dim_names

    cube = xr.DataArray(da.random.random((1, 1, 1, 1)), dims=dc_dim_names)

    mapping = d.get_datacube_dimension_mapping(cube)
    assert len(idx) == len(mapping)
    assert len(model_dim_names) == len(mapping)

    for i, model_dim_name in enumerate(model_dim_names):
        if mapping[i] is not None:
            mapped_dim_name = mapping[i][0]
            map_idx = mapping[i][1]
            assert mapped_dim_name == dc_dim_names[map_idx]


def test_get_datacube_output_dimension_mapping(mlm_item):
    mlm_item.ext.mlm.input[0].input.dim_order = ["batch", "lat", "lon", "band"]
    mlm_item.ext.mlm.output[0].result.dim_order = ["batch", "lat", "lon", "band", "foo"]
    d = DummyMLModel(mlm_item)

    dc = xr.DataArray(da.random.random((1, 1, 1, 1)), dims=["batch", "y", "x", "bands"])

    out_mapping = d.get_datacube_output_dimension_mapping(dc)
    assert out_mapping == ["batch", "y", "x", "bands", "foo"]


@pytest.mark.parametrize(
    "dc_dims, ignore_batch, valid",
    (
        (["batch", "channel", "width", "height"], True, True),
        (["asdf", "channel", "width", "height"], False, False),
        (["batch", "channel", "width", "asdf"], True, False),
        (["batch", "channel", "width", "asdf"], False, False),
    ),
)
def test_check_dimensions_present_in_datacube(
    mlm_item: pystac.Item, dc_dims: list[str], ignore_batch: bool, valid: bool
):
    d = DummyMLModel(mlm_item)
    dc = xr.DataArray(da.random.random((1, 1, 1, 1)), dims=dc_dims)

    if valid:
        d._check_dimensions_present_in_datacube(dc, ignore_batch)
    else:
        with pytest.raises(DimensionMissing):
            d._check_dimensions_present_in_datacube(dc, ignore_batch)


@pytest.mark.parametrize(
    "dc_shape, ignore_batch, valid",
    (
        ([10, 4, 224, 224], False, True),
        ([10, 4, 224, 224], True, True),
        ([10, 10, 230, 230], True, True),
        ([10, 10, 230, 230], False, True),
        ([10, 10, 230, 230], True, True),
        ([10, 2, 230, 230], True, False),
        ([10, 10, 15, 230], False, False),
    ),
)
def test_check_datacube_dimension_size(
    mlm_item: pystac.Item, dc_shape: list[int], ignore_batch: bool, valid: bool
):
    d = DummyMLModel(mlm_item)
    dc_dims = ["batch", "channel", "width", "height"]

    dc = xr.DataArray(da.random.random(dc_shape), dims=dc_dims)
    if valid:
        d._check_datacube_dimension_size(dc, ignore_batch)
    else:
        with pytest.raises(DimensionMismatch):
            d._check_datacube_dimension_size(dc, ignore_batch)


@pytest.mark.parametrize(
    "m_bands, dc_bands, dc_band_dim_name, exception",
    (
        ([], ["B02", "B03"], "band", None),
        (["B02", "B03"], ["B02", "B03"], "band", None),
        (["B02", "B03"], ["B02", "B03", "B04"], "band", None),
        (["B02", "B03"], ["B02", "B03"], "asdf", DimensionMissing),
        (["B02", "B03"], ["B02", "B04"], "band", LabelDoesNotExist),
        (
            [mlm.ModelBand({"name": "B02"}), mlm.ModelBand({"name": "B03"})],
            ["B02", "B03"],
            "band",
            None,
        ),
        (
            [mlm.ModelBand({"name": "B02"}), mlm.ModelBand({"name": "B03"})],
            ["B02", "B03", "B04"],
            "band",
            None,
        ),
        (
            [mlm.ModelBand({"name": "B02"}), mlm.ModelBand({"name": "B03"})],
            ["B02", "B04"],
            "band",
            LabelDoesNotExist,
        ),
        (
            [
                mlm.ModelBand({"name": "NDVI", "format": "asdf"}),
                mlm.ModelBand({"name": "B02"}),
            ],
            ["B02", "B04"],
            "band",
            ValueError,
        ),
        (
            [
                mlm.ModelBand({"name": "NDVI", "expression": "asdf"}),
                mlm.ModelBand({"name": "B02"}),
            ],
            ["B02", "B04"],
            "band",
            ValueError,
        ),
        (
            [
                mlm.ModelBand({"name": "B04"}),
                mlm.ModelBand({"name": "B08"}),
                mlm.ModelBand(
                    {
                        "name": "NDVI",
                        "format": "python",
                        "expression": "(B08-B04)/(B08+B04)",
                    }
                ),
            ],
            ["B04", "B08"],
            "band",
            None,
        ),
    ),
)
def test_check_datacube_bands(
    mlm_item: pystac.Item,
    m_bands: list[str | mlm.ModelBand],
    dc_bands: list[str],
    dc_band_dim_name: str,
    exception: type[Exception] | None,
):
    mlm_item.ext.mlm.input[0].bands = m_bands
    d = DummyMLModel(mlm_item)

    dc = xr.DataArray(
        da.random.random((1, 1, len(dc_bands))),
        dims=["x", "y", dc_band_dim_name],
        coords={"x": [1], "y": [1], dc_band_dim_name: dc_bands},
    )

    if exception is None:
        d._check_datacube_bands(dc)
    else:
        with pytest.raises(exception):
            d._check_datacube_bands(dc)


@pytest.mark.parametrize(
    "dc_dims, dc_dim_shp, exception_raised",
    (
        (("time", "x", "y", "bands"), (4, 1000, 1000, 8), None),
        (("x", "y", "t", "bands"), (1000, 1000, 4, 8), None),
        (("times", "x", "y", "channel"), (4, 1000, 1000, 8), None),
        (("time", "x", "y"), (4, 1000, 1000), DimensionMissing),
        (("time", "x", "y", "bands"), (4, 100, 100, 8), DimensionMismatch),
    ),
)
def test_check_datacube_dimensions(
    mlm_item: pystac.Item,
    dc_dims: list[str],
    dc_dim_shp: list[int],
    exception_raised: type[Exception],
):
    dc = xr.DataArray(da.random.random(dc_dim_shp), dims=dc_dims)

    assert len(dc_dim_shp) == len(dc_dim_shp)

    mlm_item.ext.mlm.input[0].input.shape = (-1, 1, 128, 128, 4)
    mlm_item.ext.mlm.input[0].input.dim_order = ("batch", "time", "x", "y", "bands")

    d = DummyMLModel(mlm_item)

    if exception_raised is None:
        # positive tests: should work flawlessly
        d.check_datacube_dimensions(dc, True)
    else:
        # negative test: when an exception is raised
        with pytest.raises(exception_raised):
            d.check_datacube_dimensions(dc, True)


def test_get_index_subsets(mlm_item):
    mlm_item.ext.mlm.input[0].input.dim_order = ["batch", "x", "y"]
    mlm_item.ext.mlm.input[0].input.shape = [-1, 2, 2]
    d = DummyMLModel(mlm_item)

    dc = xr.DataArray(da.random.random((5, 5, 2)), dims=["x", "y", "time"])

    idxes = list(d.get_index_subsets(dc))
    print(idxes)
    assert len(idxes) == 4
    assert (0, 0) in idxes
    assert (0, 2) in idxes
    assert (2, 0) in idxes
    assert (2, 2) in idxes


@pytest.mark.parametrize(
    "model_bands",
    (
        ["B04", "B08"],
        [mlm.ModelBand({"name": "B04"}), mlm.ModelBand({"name": "B08"})],
        ["red", "nir"],
        ["RED", "NIR"],
    ),
)
def test_select_bands(mlm_item: pystac.Item, model_bands: list[str | mlm.ModelBand]):
    dc = xr.DataArray(
        da.random.random((3, 3)),
        dims=["x", "bands"],
        coords={"x": [1, 2, 3], "bands": ["B03", "B04", "B08"]},
    )

    mlm_item.ext.mlm.input[0].bands = model_bands
    d = DummyMLModel(mlm_item)

    new_dc = d.select_bands(dc)
    assert new_dc.coords["bands"].values.tolist() == ["B04", "B08"]


def test_reorder_dc_dims_for_model_input(mlm_item: pystac.Item):
    d = DummyMLModel(mlm_item)
    dc = xr.DataArray(da.random.random((1, 1, 1)), dims=["height", "width", "channel"])

    assert dc.dims == ("height", "width", "channel")
    new_dc = d.reorder_dc_dims_for_model_input(dc)
    assert new_dc.dims == ("channel", "width", "height")


def test_reshape_dc_for_input(mlm_item: pystac.Item):
    model_input_dims = ["batch", "band", "x", "y"]
    model_input_shape = [-1, 3, 5, 5]
    mlm_item.ext.mlm.input[0].input.dim_order = model_input_dims
    mlm_item.ext.mlm.input[0].input.shape = model_input_shape

    d = DummyMLModel(mlm_item)

    dc_dims = ["b", "y", "x"]
    dc_shp = [3, 15, 15]
    dc = xr.DataArray(da.random.random(dc_shp), dims=dc_dims)

    new_dc = d.reshape_dc_for_input(dc)
    print("\n- - - - - - -")
    print(new_dc)
    print("- - - - -")


@pytest.mark.parametrize(
    "batch_recomm, batch_dim_shp, true_batch_size",
    (
        (None, None, 1),
        (5, None, 5),
        (None, 3, 3),
        (None, -1, 12),  # todo: dont hard-code fallback value
        (5, 3, 3),
        (5, 5, 5),
        (5, -1, 5),
    ),
)
def test_get_batch_size(
    mlm_item: pystac.Item, batch_recomm: int, batch_dim_shp: int, true_batch_size: int
):
    in_dims = []
    shape = []
    if batch_dim_shp is not None:
        in_dims.append("batch")
        shape.append(batch_dim_shp)

    mlm_item.ext.mlm.batch_size_suggestion = batch_recomm
    mlm_item.ext.mlm.input[0].input.dim_order = in_dims
    mlm_item.ext.mlm.input[0].input.shape = shape
    d = DummyMLModel(mlm_item)

    b_size = d.get_batch_size()
    assert b_size == true_batch_size


@pytest.mark.parametrize(
    "in_dc_dims, out_mlm_dims, out_dc_dims",
    (
        (
            ["batch", "bands", "y", "x", "time"],
            ["batch", "embedding"],
            ["batch", "embedding", "y", "x", "time"],
        ),
        (["time"], ["batch", "time"], ["batch", "time"]),
    ),
)
def test_get_output_datacube_dimensions(
    mlm_item: pystac.Item,
    in_dc_dims: list[str],
    out_mlm_dims: list[str],
    out_dc_dims: list[str],
):
    mlm_item.ext.mlm.input[0].input.dim_order = in_dc_dims
    mlm_item.ext.mlm.output[0].result.dim_order = out_mlm_dims
    d = DummyMLModel(mlm_item)

    in_dc = xr.DataArray(da.random.random([1 for _ in in_dc_dims]), dims=in_dc_dims)

    out_dc_dims_computed = d.get_output_datacube_dimensions(in_dc)
    assert out_dc_dims == out_dc_dims_computed


@pytest.mark.parametrize(
    "in_dc_dims, out_mlm_dims, diff",
    (
        (["batch", "bands", "y", "x", "time"], ["batch", "emb"], ([1], [1])),
        (["batch", "bands", "y", "x", "time"], ["batch", "emb", "foo"], ([1], [1, 2])),
        (["batch", "time"], ["batch", "time"], ([], [])),
        (["batch", "time"], ["batch", "time", "foo"], ([], [2])),
        (["batch", "time"], ["foo", "batch", "time"], ([], [0])),
    ),
)
def test_compare_input_output_dimensions(
    mlm_item: pystac.Item,
    in_dc_dims: list[str],
    out_mlm_dims: list[str],
    diff: tuple[list[int], list[int]],
):
    mlm_item.ext.mlm.input[0].input.dim_order = in_dc_dims
    mlm_item.ext.mlm.output[0].result.dim_order = out_mlm_dims
    d = DummyMLModel(mlm_item)

    in_dc = xr.DataArray(da.random.random([1 for _ in in_dc_dims]), dims=in_dc_dims)

    compare = d.compare_input_output_dimensions(in_dc)
    assert diff == compare


def test_get_chunk_output_shape(mlm_item: pystac.Item):
    in_dims = ["batch", "bands", "y", "x"]
    mlm_item.ext.mlm.input[0].input.dim_order = in_dims
    mlm_item.ext.mlm.input[0].input.shape = [-1, 12, 224, 224]

    mlm_item.ext.mlm.output[0].result.dim_order = ["batch", "emb"]
    mlm_item.ext.mlm.output[0].result.shape = [-1, 20]

    d = DummyMLModel(mlm_item)
    in_dc = xr.DataArray(
        da.random.random((10, 12, 224, 224, 3)),
        dims=["batch", "bands", "y", "x", "time"],
    )

    chunk_out_shape = d.get_chunk_output_shape(in_dc)

    assert len(chunk_out_shape) == 5
    assert chunk_out_shape == (12, 20, 1, 1, 1)  # todo dont hard-code batch size (12)


def test_get_chunk_shape(mlm_item):
    in_dims = ["batch", "bands", "y", "x"]
    mlm_item.ext.mlm.input[0].input.dim_order = in_dims
    d = DummyMLModel(mlm_item)

    in_dc = xr.DataArray(
        da.random.random((10, 12, 224, 224, 3)),
        dims=["batch", "bands", "y", "x", "time"],
    )

    chunks_shape = d.get_chunk_shape(in_dc)

    assert len(chunks_shape) == 5

    assert "batch" in chunks_shape
    assert "bands" in chunks_shape
    assert "y" in chunks_shape
    assert "x" in chunks_shape
    assert "time" in chunks_shape

    assert chunks_shape["batch"] == 12  # todo dont hard-code batch size
    assert chunks_shape["bands"] == 12
    assert chunks_shape["y"] == 224
    assert chunks_shape["x"] == 224
    assert chunks_shape["time"] == 1


def test_save_block_data(mlm_item, monkeypatch):
    mlm_item.ext.mlm.input[0].input.shape = [-1, 2, 4, 4]
    block = np.ones((2, 2, 4, 4, 1))
    d = DummyMLModel(mlm_item)

    tmp_path = "in_tmp"
    mock_np_save = unittest.mock.Mock(return_value=None)
    monkeypatch.setattr(np, "save", mock_np_save)

    out = d.save_blocks(block, tmp_path)

    assert out.shape == (1, 1)

    uuid_str = out.item().decode()
    uuid.UUID(uuid_str)
    block_filepath = f"{tmp_path}/{uuid_str}.npy"

    mock_np_save.assert_called_once()
    call_path, call_array = mock_np_save.call_args.args

    assert call_path == block_filepath
    assert np.all(call_array == np.ones((2, 2, 4, 4)))


def test_save_block_nans(mlm_item, monkeypatch):
    mlm_item.ext.mlm.input[0].input.shape = [-1, 2, 4, 4]
    block = np.full((2, 2, 4, 4, 1), np.nan)

    mock_save_np = unittest.mock.Mock(return_value=None)
    monkeypatch.setattr(np, "save", mock_save_np)

    d = DummyMLModel(mlm_item)
    out = d.save_blocks(block, "foo")

    mock_save_np.assert_not_called()
    assert out.shape == (1, 1)

    uuid_str = out.item().decode()
    assert uuid_str == "00000000-0000-0000-0000-000000000000"


def test_load_prediction(mlm_item, monkeypatch):
    d = DummyMLModel(mlm_item)

    tmp_dir = "tmp_out"

    mock_np_load = unittest.mock.Mock(return_value=np.array([1, 2]))
    monkeypatch.setattr(np, "load", mock_np_load)

    block = np.array([[b"asdf"]])

    loaded_block = d.load_prediction(block, tmp_dir, 2, None)

    mock_np_load.assert_called_once_with(f"{tmp_dir}/asdf.npy")

    assert loaded_block.shape == (2, 1, 1, 1)
    assert np.all(loaded_block == np.array([[[[1]]], [[[2]]]]))


def test_load_prediction_nans(mlm_item, monkeypatch):
    d = DummyMLModel(mlm_item)
    batch_count = 12  # todo: dont hard-code batch count

    mock_np_load = unittest.mock.Mock(return_value=np.ones((1, 2)))
    monkeypatch.setattr(np, "load", mock_np_load)

    block = np.array([[b"00000000-0000-0000-0000-000000000000"]])
    loaded_block = d.load_prediction(block, "foo", 2, None)

    mock_np_load.assert_not_called()

    out_dims = mlm_item.ext.mlm.output[0].result.dim_order
    out_shp = mlm_item.ext.mlm.output[0].result.shape
    out_shp[out_dims.index("batch")] = batch_count
    out_shp.extend([1, 1, 1])
    out_shp = tuple(out_shp)

    assert loaded_block.shape == out_shp  # this fails if the batch size changes
    assert np.isnan(loaded_block).all()


@pytest.mark.parametrize("exec_mode", ("dask", "subprocess"))
def test_predict_in_dask_worker(mlm_item, monkeypatch, exec_mode):
    d = DummyMLModel(mlm_item)

    file_returns = [Path("a.npy"), Path("b.npy")]
    mock_list_files = unittest.mock.Mock(return_value=file_returns)
    monkeypatch.setattr(Path, "glob", mock_list_files)

    mock_make_predictions = unittest.mock.Mock(return_value=True)
    monkeypatch.setattr(DummyMLModel, "make_predictions", mock_make_predictions)

    mock_subprocess_run = unittest.mock.Mock(
        return_value=subprocess.CompletedProcess(args="foo", returncode=0)
    )
    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    monkeypatch.setattr(d, "_model_filepath", "asdf.model")
    pre_func = pystac.extensions.mlm.ProcessingExpression.create("python", "pre-func")
    post_func = pystac.extensions.mlm.ProcessingExpression.create("python", "post-func")
    monkeypatch.setattr(d.input, "pre_processing_function", pre_func)
    monkeypatch.setattr(d.output, "post_processing_function", post_func)

    in_dir = "in_dir"
    out_dir = "out_dir"

    if exec_mode == "dask":
        out = d.predict_in_dask_worker(in_dir, out_dir, None)
    elif exec_mode == "subprocess":
        out = d.predict_in_subprocess(in_dir, out_dir, None)
    else:
        raise NotImplementedError("this should not be reached")

    # mock methods not called yet as this is a dask delayed object
    assert isinstance(out, Delayed)
    mock_list_files.assert_not_called()
    mock_make_predictions.assert_not_called()
    mock_subprocess_run.assert_not_called()

    # after compute the mock methods have been called
    out = out.compute()
    if exec_mode == "dask":
        mock_list_files.assert_called_once()
        mock_subprocess_run.assert_not_called()
        mock_make_predictions.assert_called_once_with(
            "asdf.model", file_returns, Path(out_dir), pre_func, post_func
        )
    elif exec_mode == "subprocess":
        mock_make_predictions.assert_not_called()
        mock_subprocess_run.assert_called_once_with(
            [
                in_dir,
                out_dir,
                "--preprocessing_function",
                pre_func.expression,
                "--postprocessing_function",
                post_func.expression,
            ]
        )

    assert isinstance(out, bool)
    assert out is True


def test_reorder_out_dc_dims(mlm_item: pystac.Item):
    in_dc = xr.DataArray(
        da.random.random((1, 1, 1, 1)), dims=["time", "bands", "y", "x"]
    )

    out_dc = xr.DataArray(
        da.random.random((1, 1, 1, 1)), dims=["embedding", "x", "y", "time"]
    )

    d = DummyMLModel(mlm_item)
    reordered = d.reorder_out_dc_dims(in_dc, out_dc)
    assert len(reordered.dims) == 4
    assert reordered.dims == ("time", "embedding", "y", "x")
