"""
We overwrite the implementation of openeo-processes-dask aggregate_spatial
implementation for two reasons:
1) usage of xvec.zonal with method=iterate explodes RAM usage used on many polygons.
   Solution: Use method=rasterize
2) xvec implementation of of zonal statistic with method rasteirze seems buggy:
   Solution: use dtype=float32 (already fixed in later version on GH)
"""

import gc
import logging
from collections.abc import Hashable, Sequence
from typing import Callable

import dask_geopandas as d_gpd
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)
from xarray import Dataset

logger = logging.getLogger(__name__)


def _merge_dicts(d1: dict[str, list], d2: dict[str, int | float | str]):
    current_len = len(next(iter(d1.values()))) if d1 else 0

    # 2. Iterate through the union of all keys
    for key in set(d1.keys()) | set(d2.keys()):
        if key in d1:
            # If key is in d1, append the value from d2 (or None if missing)
            d1[key].append(d2.get(key))
        else:
            # If key is only in d2, create the list with Nones + the d2 value
            d1[key] = [None] * current_len + [d2[key]]


def _geojson_parse_feature(feature: dict) -> dict[str, int | float | str]:
    if feature["type"] != "Feature":
        raise TypeError('The dictionary passed in as feature is not of type "Feature"')

    # holds the featuer's properties without lists or dicts, e.g. {"name": value}
    props = {}
    if "properties" not in feature:
        return props

    for prop_k in feature["properties"]:
        prop_v = feature["properties"][prop_k]

        # filter out dict or list properties
        if isinstance(prop_v, dict) or isinstance(prop_v, list):
            continue

        props[prop_k] = prop_v

    return props


def _geojson_parse_featurecollection(collection: dict):
    if collection["type"] != "FeatureCollection":
        raise ValueError(
            'The dictionary passed in as collection is not of type "FeatureCollection"'
        )

    # holds the collection's props, e.g. {"name": [val1, val2, None, val3, ...}
    feature_collection_props = {}

    for feature in collection["features"]:
        feature_props = _geojson_parse_feature(feature)
        _merge_dicts(feature_collection_props, feature_props)

    return feature_collection_props


def _geojson_parse_geojson(geometries: dict) -> dict[str, list]:
    if "type" not in geometries:
        return {}

    if geometries["type"] == "FeatureCollection":
        return _geojson_parse_featurecollection(geometries)

    if geometries["type"] == "Feature":
        feature_props = _geojson_parse_feature(geometries)
        return {k: [feature_props[k]] for k in feature_props}

    else:
        # case for type= any geometry (Polygon, Line, etc.)
        return {}


def _agg_rasterize(groups, stats, **kwargs):
    if isinstance(stats, str):
        return getattr(groups, stats)(**kwargs)
    return groups.reduce(stats, keep_attrs=True, **kwargs)


def _zonal_stats_rasterize_new(
    acc,
    geometry: Sequence[shapely.Geometry],
    x_coords: Hashable,
    y_coords: Hashable,
    stats: str | Callable | Sequence[str | Callable | tuple] = "mean",
    name: str = "geometry",
    all_touched: bool = False,
    **kwargs,
):
    try:
        import rasterio
        import rioxarray  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "The rioxarray package is required for `zonal_stats()`. "
            "You can install it using 'conda install -c conda-forge rioxarray' or "
            "'pip install rioxarray'."
        ) from err

    if hasattr(geometry, "crs"):
        crs = geometry.crs
    else:
        crs = None

    transform = acc._obj.rio.transform()

    labels = rasterio.features.rasterize(
        zip(geometry, range(len(geometry))),
        out_shape=(
            acc._obj[y_coords].shape[0],
            acc._obj[x_coords].shape[0],
        ),
        transform=transform,
        fill=np.nan,
        all_touched=all_touched,
        dtype=np.float32,
    )
    groups = acc._obj.groupby(xr.DataArray(labels, dims=(y_coords, x_coords)))

    if pd.api.types.is_list_like(stats):
        agg = {}
        for stat in stats:
            if isinstance(stat, str):
                agg[stat] = _agg_rasterize(groups, stat, **kwargs)
            elif callable(stat):
                agg[stat.__name__] = _agg_rasterize(groups, stat, **kwargs)
            elif isinstance(stat, tuple):
                kws = stat[2] if len(stat) == 3 else {}
                agg[stat[0]] = _agg_rasterize(groups, stat[1], **kws)
            else:
                raise ValueError(f"{stat} is not a valid aggregation.")

        agg = xr.concat(
            agg.values(),
            dim=xr.DataArray(
                list(agg.keys()), name="zonal_statistics", dims="zonal_statistics"
            ),
        )
    elif isinstance(stats, str) or callable(stats):
        agg = _agg_rasterize(groups, stats, **kwargs)
    else:
        raise ValueError(f"{stats} is not a valid aggregation.")

    vec_cube = (
        agg.reindex(group=range(len(geometry)))
        .assign_coords(group=geometry)
        .rename(group=name)
    ).xvec.set_geom_indexes(name, crs=crs)

    del groups
    gc.collect()

    return vec_cube


def aggregate_spatial(
    data: RasterCube,
    geometries,
    reducer: Callable,
    chunk_size: int = 2,
) -> VectorCube:
    # monkey-patch xvec.zonal._zonal_stats_rasterize
    from unittest.mock import patch

    DEFAULT_CRS = "EPSG:4326"
    x_dim = data.openeo.x_dim
    y_dim = data.openeo.y_dim

    if isinstance(geometries, dict):
        # if its a dict, that means we are dealing with a geojson
        new_coords = _geojson_parse_geojson(geometries)

        if "features" in geometries:
            for feature in geometries["features"]:
                if "properties" not in feature:
                    feature["properties"] = {}
                elif feature["properties"] is None:
                    feature["properties"] = {}
            if isinstance(geometries.get("crs", {}), dict):
                DEFAULT_CRS = (
                    geometries.get("crs", {})
                    .get("properties", {})
                    .get("name", DEFAULT_CRS)
                )
            else:
                DEFAULT_CRS = int(geometries.get("crs", {}))
            logger.info(f"CRS in geometries: {DEFAULT_CRS}.")

            if "type" in geometries and geometries["type"] == "FeatureCollection":
                gdf = gpd.GeoDataFrame.from_features(geometries, crs=DEFAULT_CRS)
            elif "type" in geometries and geometries["type"] in ["Polygon"]:
                polygon = shapely.geometry.Polygon(geometries["coordinates"][0])
                gdf = gpd.GeoDataFrame(geometry=[polygon])
                gdf.crs = DEFAULT_CRS

    elif isinstance(geometries, gpd.GeoDataFrame):
        raise NotImplementedError(
            "Not Implemented, Provide geometries directly as geojson."
        )
    elif isinstance(geometries, d_gpd.GeoDataFrame):
        raise NotImplementedError(
            "Not Implemented, Provide geometries directly as geojson."
        )
    elif isinstance(geometries, Dataset):
        raise NotImplementedError(
            "Not Implemented, Provide geometries directly as geojson."
        )
    else:
        raise NotImplementedError(
            "Not Implemented, Provide geometries directly as geojson."
        )

    gdf = gdf.to_crs(data.rio.crs)
    geometries = gdf.geometry.values

    positional_parameters = {"data": 0}

    with patch("xvec.accessor._zonal_stats_rasterize", new=_zonal_stats_rasterize_new):
        vec_cube = data.xvec.zonal_stats(
            geometries,
            x_coords=x_dim,
            y_coords=y_dim,
            method="rasterize",
            stats=reducer,
            positional_parameters=positional_parameters,
        )

    # we will add geometry properties as non-dimensional coordinates to the vector data cube

    # Order of geometries in geom column is the same as in input geometry sequence
    # https://xvec.readthedocs.io/en/stable/generated/xarray.Dataset.xvec.zonal_stats.html
    # therefore we can assign the properties as additional coordinates to geometry
    # dimension in the same order
    vector_cube = vec_cube.assign_coords(
        **{param: ("geometry", new_coords[param]) for param in new_coords}
    )

    return vector_cube
