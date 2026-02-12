import json
import os
from pathlib import Path

if Path.cwd().resolve().name == "examples":
    os.chdir("..")
if Path.cwd().resolve().name != "openeo-processes-dask-ml":
    raise Exception("Current CWD is not the Project root (openeo-processes-dask-ml)")


from minibackend import execute_graph_dict

# mlm_class_random_forest("sqrt", 50)

with open("examples/training_data/train_data_small.json") as file:
    geoms = json.load(file)


process_graph = {
    "process_graph": {
        "load_data": {
            "process_id": "load_stac",
            "arguments": {
                "url": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a",
                "spatial_extent": {
                    "west": -2.96,
                    "east": -2.7,
                    "south": 47.91,
                    "north": 48.06,
                },
                "temporal_extent": ["2022-01-01", "2022-02-28"],
                "bands": [
                    # "coastal",
                    # "blue",
                    # "green",
                    "red",
                    # "rededge1",
                    # "rededge2",
                    # "rededge3",
                    "nir",
                    # "nir08",
                    # "nir09",
                    # "swir16",
                    # "swir22",
                ],
                "resolution": 10,
            },
        },
        "agg_temp": {
            "process_id": "aggregate_temporal_period",
            "arguments": {
                "data": {"from_node": "load_data"},
                "period": "month",
                "reducer": {
                    "process_graph": {
                        "median1": {
                            "process_id": "median",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
        },
        "ndvi": {
            "process_id": "ndvi",
            "arguments": {"data": {"from_node": "agg_temp"}, "target_band": "NDVI"},
        },
        "training_data": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "ndvi"},
                "geometries": geoms,
                "reducer": {
                    "process_graph": {
                        "median1": {
                            "process_id": "median",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
        },
        "rf_init": {
            "process_id": "mlm_class_random_forest",
            "arguments": {"max_variables": "sqrt", "num_trees": 50},
        },
        "fit": {
            "process_id": "ml_fit",
            "arguments": {
                "model": {"from_node": "rf_init"},
                "training_set": {"from_node": "training_data"},
                "target": "class_name",
            },
            "result": True,
        },
    },
    "parameters": [],
}

out = execute_graph_dict(process_graph)
print(out)
