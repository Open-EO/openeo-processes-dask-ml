from typing import Callable

from openeo_processes_dask.process_implementations.cubes.aggregate import (
    aggregate_spatial as aggregate_spatial_original,
)
from openeo_processes_dask.process_implementations.data_model import (
    RasterCube,
    VectorCube,
)


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


def aggregate_spatial(
    data: RasterCube,
    geometries,
    reducer: Callable,
    chunk_size: int = 2,
) -> VectorCube:
    print("my own aggregate_spatial implementation")

    vector_cube = aggregate_spatial_original(data, geometries, reducer, chunk_size)

    # we will add geometry properties as auxilary coordinates to the vector data cube

    if isinstance(geometries, dict):
        # if its a dict, that means we are dealing with a geojson
        new_coords = _geojson_parse_geojson(geometries)
    else:
        new_coords = {}

    # Order of geometries in geom column is the same as in input geometry sequence
    # https://xvec.readthedocs.io/en/stable/generated/xarray.Dataset.xvec.zonal_stats.html
    # therefore we can assign the properties as additional coordinates to geometry
    # dimension in the same order
    vector_cube = vector_cube.assign_coords(
        **{param: ("geometry", new_coords[param]) for param in new_coords}
    )

    return vector_cube
