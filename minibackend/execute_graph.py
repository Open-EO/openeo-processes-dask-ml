import logging
import sys

from openeo_pg_parser_networkx import OpenEOProcessGraph

from .openeo_minibackend import process_registry


### ------ Configure logger: DEBUG and INFO go to stdout, WARNING and above to stderr
class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        # Allow only log records at or below the maximum level
        return record.levelno <= self.max_level


# Formatter to style log messages
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)

# 1. Standard Output Handler (DEBUG & INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)  # Catch DEBUG and above...
stdout_handler.addFilter(
    MaxLevelFilter(logging.INFO)
)  # ...but filter out WARNING, ERROR, CRITICAL
stdout_handler.setFormatter(formatter)

# 2. Standard Error Handler (WARNING, ERROR, CRITICAL)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)  # Catch WARNING and above
stderr_handler.setFormatter(formatter)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Ensure root logger passes all levels to handlers

# Add the handlers
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)

logging.debug("This goes to STDOUT")
logging.info("This goes to STDOUT")
logging.warning("This goes to STDERR")
logging.error("This goes to STDERR")
logging.critical("This goes to STDERR")


def execute_graph_file(path: str):
    parsed_graph = OpenEOProcessGraph.from_file(path)
    pg_callable = parsed_graph.to_callable(process_registry=process_registry)
    r = pg_callable()
    return r


def execute_graph_dict(graph: dict):
    parsed_graph = OpenEOProcessGraph(graph)
    pg_callable = parsed_graph.to_callable(process_registry=process_registry)
    r = pg_callable()
    return r
