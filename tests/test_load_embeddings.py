from datetime import datetime

import dask.array as da
import numpy as np
import pyproj
import pystac
import pytest
import shapely
import xarray as xr
from dask import array as da
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

from openeo_processes_dask_ml.process_implementations import load_embeddings


def test_get_item_time():
    dt = datetime.fromisoformat("2026-07-15T02:54:14-12:00")
    start_dt = datetime.fromisoformat("2015-07-15T02:54:14-12:00")
    end_dt = datetime.fromisoformat("2015-07-15T02:54:14-12:00")

    i = pystac.Item("asdf", None, None, dt, {})
    t = load_embeddings._get_item_time(i)
    assert isinstance(t, datetime)
    assert t.isoformat() == dt.isoformat()

    i = pystac.Item(
        "asdf", None, None, dt, {}, start_datetime=start_dt, end_datetime=end_dt
    )
    t = load_embeddings._get_item_time(i)
    assert isinstance(t, datetime)
    assert t.isoformat() == dt.isoformat()

    i = pystac.Item(
        "asdf", None, None, None, {}, start_datetime=start_dt, end_datetime=end_dt
    )
    t = load_embeddings._get_item_time(i)
    assert isinstance(t, datetime)
    assert t.isoformat() == start_dt.isoformat()


def test_match_geom_in_list_point():
    pnt1 = shapely.Point([5.555, 1.111])
    pnt2 = shapely.Point([5.555, 1.111])
    pnt3 = shapely.Point([10.3, 3.3])
    pnt4 = shapely.Point([10.3, 3.300001])

    geom_list = []

    i = load_embeddings._match_geom_in_list(geom_list, pnt1, tolerance=0.00001)
    assert i is None
    geom_list.append(pnt1)

    i = load_embeddings._match_geom_in_list(geom_list, pnt2, tolerance=0.00001)
    assert i == 0

    i = load_embeddings._match_geom_in_list(geom_list, pnt3, tolerance=0.00001)
    assert i is None
    geom_list.append(pnt3)

    i = load_embeddings._match_geom_in_list(geom_list, pnt4, tolerance=0.00001)
    assert i == 1


def test_match_geom_in_list_polygon():
    # Helper to create a small square polygon offset by a specific coordinate
    def make_square(x, y):
        # Creates a 1x1 square with the bottom-left corner at (x, y)
        return shapely.Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])

    # Recreate the exact same spatial layout/relationships using Polygons
    poly1 = make_square(5.555, 1.111)
    poly2 = make_square(5.555, 1.111)
    poly3 = make_square(10.3, 3.3)
    poly4 = make_square(10.3, 3.300001)

    geom_list = []

    # 1. First polygon shouldn't match anything in an empty list
    i = load_embeddings._match_geom_in_list(geom_list, poly1, tolerance=0.00001)
    assert i is None
    geom_list.append(poly1)

    # 2. Second polygon is identical to the first, so it should match index 0
    i = load_embeddings._match_geom_in_list(geom_list, poly2, tolerance=0.00001)
    assert i == 0

    # 3. Third polygon is in a new location, shouldn't match index 0
    i = load_embeddings._match_geom_in_list(geom_list, poly3, tolerance=0.00001)
    assert i is None
    geom_list.append(poly3)

    # 4. Fourth polygon is offset within the 0.00001 tolerance, so it should match index 1
    i = load_embeddings._match_geom_in_list(geom_list, poly4, tolerance=0.00001)
    assert i == 1


def test_load_tiff_1x1():
    path = "tests/data/embedding_1x1.tif"
    e_dc = load_embeddings._load_tiff(path)

    assert "band" not in e_dc.dims
    assert "embedding" in e_dc.dims
    assert "x" not in e_dc.dims
    assert "y" not in e_dc.dims
    assert isinstance(e_dc.data, da.Array)

    e_dc.compute()


def test_prepare_geoparquet_without_bbox_without_transform():
    path = "tests/data/embeddings.parquet"
    geom_array, da_array = load_embeddings._prepare_geoparquet(
        path, None, "geometry", "embedding", 4, np.float32, False
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 4
    assert da_array.shape == (4, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = poly_utm.contains(geom_array)
    outside = ~poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_prepare_geoparquet_without_bbox_with_transform():
    path = "tests/data/embeddings.parquet"
    geom_array, da_array = load_embeddings._prepare_geoparquet(
        path, None, "geometry", "embedding", 4, np.float32, True
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 4
    assert da_array.shape == (4, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = ~poly_utm.contains(geom_array)
    outside = poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_prepare_geoparquet_with_bbox_without_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    geom_array, da_array = load_embeddings._prepare_geoparquet(
        path, bbox, "geometry", "embedding", 4, np.float32, False
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 2
    assert da_array.shape == (2, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = poly_utm.contains(geom_array)
    outside = ~poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_prepare_geoparquet_with_bbox_with_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    geom_array, da_array = load_embeddings._prepare_geoparquet(
        path, bbox, "geometry", "embedding", 4, np.float32, True
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 2
    assert da_array.shape == (2, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = ~poly_utm.contains(geom_array)
    outside = poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_load_parquet_item_without_bbox_without_transform():
    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = load_embeddings._load_parquet_item(path, None, False)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (4, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    inside = poly_utm.contains(geom_coords)
    outside = ~poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:32632"))


def test_load_parquet_item_without_bbox_with_transform():
    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = load_embeddings._load_parquet_item(path, None, True)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (4, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    outside = ~poly_utm.contains(geom_coords)
    inside = poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:4326"))


def test_load_parquet_item_with_bbox_without_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = load_embeddings._load_parquet_item(path, bbox, False)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (2, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    inside = poly_utm.contains(geom_coords)
    outside = ~poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:32632"))


def test_load_parquet_item_with_bbox_with_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = load_embeddings._load_parquet_item(path, bbox, True)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (2, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    outside = ~poly_utm.contains(geom_coords)
    inside = poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:4326"))


def test_load_embedding_item_tiff():
    # 1px, 2px
    # with and without bbox, different CRS
    # geom in returning DC must be 4326
    pass


def test_load_embedding_item_parquet():
    # with and without bbox in different CRS
    # CRS of embeddings must be unchanged
    pass


def test_construct_embedding_vector_cube():
    x = xr.DataArray(da.random.random(4))
    i = [[x, x], [x, x], [x, xr.DataArray()]]
    geoms = [shapely.Point(0, 0), shapely.Point(1, 1)]
    times = [
        datetime.fromisoformat("2026-01-01"),
        datetime.fromisoformat("2026-01-02"),
        datetime.fromisoformat("2026-01-03"),
    ]

    e_dc = load_embeddings._construct_embedding_vector_cube(i, geoms, times)

    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert e_dc.shape == (3, 2, 4)
    assert isinstance(e_dc.data, da.Array)
    assert np.isnan(
        e_dc.isel(time=2, geometry=1).compute()
    ).all()  # check nans are preserved
    assert not np.isnan(
        e_dc.isel(time=1, geometry=1).compute()
    ).all()  # check other is not nan


def test_crs_of():
    path = "tests/data/embeddings.parquet"
    s = load_embeddings._crs_of(path)
    assert s is not None
    assert isinstance(s, str)

    path = "tests/data/non-geoparquet.parquet"
    with pytest.raises(Exception):
        load_embeddings._crs_of(path)
