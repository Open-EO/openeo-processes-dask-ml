"""
This is basic example of an openEO process graph.
It shows how to classify AEF embeddings to map crop types in Breizh, France.
This is the same as in examples/process_graphs/classify_embeddings.json
"""
import json
import os
from pathlib import Path

if Path.cwd().resolve().name == "examples":
    os.chdir("..")
if Path.cwd().resolve().name != "openeo-processes-dask-ml":
    raise Exception("Current CWD is not the Project root (openeo-processes-dask-ml)")


from minibackend import execute_graph_dict

with open("examples/training_data/cornbelt_yield_per_county_2020.json") as file:
    geoms = json.load(file)


process_graph = {
    # init random forest
    "mlmclassrandomforest1": {
        "process_id": "mlm_regr_random_forest",
        "arguments": {
            "max_variables": "onethird",
            "num_trees": 200,
            "seed": 42,
            "dimension": "embedding",
            "use_timeseries": False,
        },
    },
    # 2) datacube for training
    "load_embeddings_train": {
        "process_id": "load_embeddings",
        "arguments": {"url": "examples/embeddings/terramind_cornbelt_embeddings.json"},
    },
    "subset_2020": {
        "process_id": "filter_temporal",
        "arguments": {
            "data": {"from_node": "load_embeddings_train"},
            "extent": ["2019-12-31", "2021-01-01"],
        },
    },
    "sample_values": {
        "process_id": "sample_values_at_locations",
        "arguments": {"data": {"from_node": "subset_2020"}, "geometries": geoms},
    },
    "mlfit1": {
        "process_id": "ml_fit",
        "arguments": {
            "model": {"from_node": "mlmclassrandomforest1"},
            "target": "yield_bu_per_acre",
            "training_set": {"from_node": "sample_values"},
        },
    },
    "save_ml": {
        "process_id": "save_ml_model",
        "arguments": {"data": {"from_node": "mlfit1"}, "name": "Emb regressor"},
        "result": True,
    },
}

out = execute_graph_dict(process_graph)
out = out.compute()
print(out)
