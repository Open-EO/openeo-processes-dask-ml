# from openeo_processes_dask_ml.process_implementations.overrides import aggregate
# import pytest
#
#
# @pytest.fixture
# def feature() -> dict:
#     return {
#         "type": "Feature",
#         "properties": {"a": 1, "b": 2, "c": [3, 4], "d": {1: 1}},
#         "geometry": {"type": "Point", "coordinates": [4, 5]},
#     }
#
#
# @pytest.fixture
# def feature_collection() -> dict:
#     return {
#         "type": "FeatureCollection",
#         "features": [
#             {
#                 "type": "Feature",
#                 "properties": {"a": 1, "b": 2, "c": [3, 4], "d": {1: 1}},
#                 "geometry": {"type": "Point", "coordinates": [4, 5]},
#             },
#             {
#                 "type": "Feature",
#                 "properties": {"a": 2, "c": "foo", "d": {1: 1}},
#                 "geometry": {"type": "Point", "coordinates": [4, 5]},
#             },
#             {
#                 "type": "Feature",
#                 "properties": {"a": 3, "b": 4, "d": {1: 1}, "e": "lol"},
#                 "geometry": {"type": "Point", "coordinates": [4, 5]},
#             },
#         ],
#     }
#
#
# def test_merge_dicts():
#     small_dict_1 = {"a": 1, "b": "hi"}
#     large_dict = {}
#
#     aggregate._merge_dicts(large_dict, small_dict_1)
#
#     assert "a" in large_dict
#     assert "b" in large_dict
#     assert large_dict["a"] == [1]
#     assert large_dict["b"] == ["hi"]
#
#     small_dict_2 = {"a": 2, "b": "foo"}
#     aggregate._merge_dicts(large_dict, small_dict_2)
#     assert large_dict["a"] == [1, 2]
#     assert large_dict["b"] == ["hi", "foo"]
#
#     small_dict_3 = {"b": "bar", "c": 3.14}
#     aggregate._merge_dicts(large_dict, small_dict_3)
#     assert "c" in large_dict
#     assert large_dict["a"] == [1, 2, None]
#     assert large_dict["b"] == ["hi", "foo", "bar"]
#     assert large_dict["c"] == [None, None, 3.14]
#
#
# def test_parse_geojson_feature(feature):
#     d = aggregate._geojson_parse_feature(feature)
#     assert "a" in d
#     assert d["a"] == 1
#
#     assert "b" in d
#     assert d["b"] == 2
#
#     assert "c" not in d
#     assert "d" not in d
#
#
# def test_parse_feature_collection(feature_collection):
#
#     d = aggregate._geojson_parse_featurecollection(feature_collection)
#
#     assert "a" in d
#     assert d["a"] == [1, 2, 3]
#
#     assert "b" in d
#     assert d["b"] == [2, None, 4]
#
#     assert "c" in d
#     assert d["c"] == [None, "foo", None]
#
#     assert "d" not in d
#
#     assert "e" in d
#     assert d["e"] == [None, None, "lol"]
#
#
# def test_parse_geojson(feature, feature_collection):
#     d = aggregate._geojson_parse_geojson(feature)
#     assert "a" in d
#     assert d["a"] == [1]
#
#     assert "b" in d
#     assert d["b"] == [2]
#
#     d = aggregate._geojson_parse_geojson(feature_collection)
#     assert "a" in d
#     assert d["a"] == [1, 2, 3]
#
#     assert "b" in d
#     assert d["b"] == [2, None, 4]
#
#     assert "c" in d
#     assert d["c"] == [None, "foo", None]
#
#     assert "d" not in d
#
#     assert "e" in d
#     assert d["e"] == [None, None, "lol"]
