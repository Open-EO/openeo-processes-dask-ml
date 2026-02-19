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
    "loadcollection1": {
        "process_id": "load_collection",
        "arguments": {
            "bands": ["blue", "green", "red", "nir"],
            "id": "sentinel-2-l2a",
            "spatial_extent": {
                "west": -2.997065,
                "south": 47.892255,
                "east": -2.671039,
                "north": 48.047242,
                "crs": 4326,
            },
            "temporal_extent": ["2017-03-01", "2017-05-30"],
        },
    },
    "aggregatetemporalperiod1": {
        "process_id": "aggregate_temporal_period",
        "arguments": {
            "data": {"from_node": "loadcollection1"},
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
    "ndvi1": {
        "process_id": "ndvi",
        "arguments": {
            "data": {"from_node": "aggregatetemporalperiod1"},
            "nir": "nir",
            "red": "red",
            "target_band": "NDVI",
        },
    },
    "arrayinterpolatelinear1": {
        "process_id": "array_interpolate_linear",
        "arguments": {"data": {"from_node": "ndvi1"}},
    },
    "mlmclassrandomforest1": {
        "process_id": "mlm_class_random_forest",
        "arguments": {"max_variables": "sqrt", "num_trees": 200, "seed": 42},
    },
    "loadcollection2": {
        "process_id": "load_collection",
        "arguments": {
            "bands": ["green", "blue", "red", "nir"],
            "id": "sentinel-2-l2a",
            "spatial_extent": {
                "west": -4.026661,
                "south": 48.202738,
                "east": -3.743587,
                "north": 48.300371,
                "crs": 4326,
            },
            "temporal_extent": ["2017-03-01", "2017-05-30"],
        },
    },
    "aggregatetemporalperiod2": {
        "process_id": "aggregate_temporal_period",
        "arguments": {
            "data": {"from_node": "loadcollection2"},
            "period": "month",
            "reducer": {
                "process_graph": {
                    "median2": {
                        "process_id": "median",
                        "arguments": {"data": {"from_parameter": "data"}},
                        "result": True,
                    }
                }
            },
        },
    },
    "ndvi2": {
        "process_id": "ndvi",
        "arguments": {
            "data": {"from_node": "aggregatetemporalperiod2"},
            "nir": "nir",
            "red": "red",
            "target_band": "NDVI",
        },
    },
    "arrayinterpolatelinear2": {
        "process_id": "array_interpolate_linear",
        "arguments": {"data": {"from_node": "ndvi2"}},
    },
    "aggregatespatial1": {
        "process_id": "aggregate_spatial",
        "arguments": {
            "data": {"from_node": "arrayinterpolatelinear2"},
            "geometries": geoms,
            "reducer": {
                "process_graph": {
                    "mean1": {
                        "process_id": "mean",
                        "arguments": {"data": {"from_parameter": "data"}},
                        "result": True,
                    }
                }
            },
        },
    },
    "mlfit1": {
        "process_id": "ml_fit",
        "arguments": {
            "model": {"from_node": "mlmclassrandomforest1"},
            "target": "class_name",
            "training_set": {"from_node": "aggregatespatial1"},
        },
    },
    "mlpredict1": {
        "process_id": "ml_predict",
        "arguments": {
            "data": {"from_node": "arrayinterpolatelinear1"},
            "model": {"from_node": "mlfit1"},
        },
    },
    "saveresult1": {
        "process_id": "save_result",
        "arguments": {
            "data": {"from_node": "mlpredict1"},
            "format": "GTiff",
            "options": {},
        },
        "result": True,
    },
}


out = execute_graph_dict(process_graph)

# out = out.compute()

print(out)
