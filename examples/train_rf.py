import json
import os
from pathlib import Path

from matplotlib import pyplot as plt
from xarray import DataArray

if Path.cwd().resolve().name == "examples":
    os.chdir("..")
if Path.cwd().resolve().name != "openeo-processes-dask-ml":
    raise Exception("Current CWD is not the Project root (openeo-processes-dask-ml)")


from minibackend import execute_graph_dict

with open("examples/training_data/train_data.json") as file:
    geoms = json.load(file)


process_graph = {
    # 1) datacube for prediction
    "loadcollection1": {
        "process_id": "load_collection",
        "arguments": {
            "bands": [
                "coastal",
                "blue",
                "green",
                "red",
                "rededge1",
                "rededge2",
                "rededge3",
                "nir",
                "nir08",
                "nir09",
                "swir16",
                "swir22",
                "scl",
            ],
            "id": "sentinel-2-l2a",
            "spatial_extent": {
                "west": -2.96,
                "south": 47.91,
                "east": -2.7,
                "north": 48.06,
                "crs": 4326,
            },
            "temporal_extent": ["2017-05-01T00:00:00Z", "2017-09-30T23:59:59Z"],
            "properties": {
                "eo:cloud_cover": {
                    "process_graph": {
                        "lte1": {
                            "process_id": "lte",
                            "arguments": {"x": {"from_parameter": "value"}, "y": 50},
                            "result": True,
                        }
                    }
                }
            },
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
    # "arrayinterpolatelinear1": {
    #     "process_id": "array_interpolate_linear",
    #     "arguments": {"data": {"from_node": "aggregatetemporalperiod1"}},
    # },
    "ndvi1": {
        "process_id": "ndvi",
        "arguments": {
            "data": {"from_node": "aggregatetemporalperiod1"},
            "nir": "nir",
            "red": "red",
            "target_band": "NDVI",
        },
    },
    # init the RF model
    "mlmclassrandomforest1": {
        "process_id": "mlm_class_random_forest",
        "arguments": {"max_variables": "sqrt", "num_trees": 200, "seed": 42},
    },
    # 2) datacube for training
    "loadcollection2": {
        "process_id": "load_collection",
        "arguments": {
            "bands": [
                "coastal",
                "blue",
                "green",
                "red",
                "rededge1",
                "rededge2",
                "rededge3",
                "nir",
                "nir08",
                "nir09",
                "swir16",
                "swir22",
                "scl",
            ],
            "id": "sentinel-2-l2a",
            "spatial_extent": {
                "west": -4.02,
                "south": 48.20,
                "east": -3.74,
                "north": 48.30,
                "crs": 4326,
            },
            "temporal_extent": ["2017-05-01T00:00:00Z", "2017-09-30T23:59:59Z"],
            "properties": {
                "eo:cloud_cover": {
                    "process_graph": {
                        "lte1": {
                            "process_id": "lte",
                            "arguments": {"x": {"from_parameter": "value"}, "y": 50},
                            "result": True,
                        }
                    }
                }
            },
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
    # "arrayinterpolatelinear2": {
    #     "process_id": "array_interpolate_linear",
    #     "arguments": {"data": {"from_node": "aggregatetemporalperiod2"}},
    # },
    "ndvi2": {
        "process_id": "ndvi",
        "arguments": {
            "data": {"from_node": "aggregatetemporalperiod2"},
            "nir": "nir",
            "red": "red",
            "target_band": "NDVI",
        },
    },
    "aggregatespatial1": {
        "process_id": "aggregate_spatial",
        "arguments": {
            "data": {"from_node": "ndvi2"},
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
            "data": {"from_node": "ndvi1"},
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

out: DataArray = execute_graph_dict(process_graph)
out.compute()
