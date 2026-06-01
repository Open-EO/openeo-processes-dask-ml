import json
import os
from typing import Any

import requests


def _load_stac_from_remote(uri: str) -> dict[str, Any]:
    # fetch STAC Item
    r = requests.get(uri)
    if r.status_code != 200:
        raise requests.exceptions.HTTPError(
            "Error while fetching STAC Item from URI: "
            "Server did not respond with status code 200"
        )

    try:
        stac = r.json()
    except requests.exceptions.JSONDecodeError:
        raise Exception("The provided URI does not point to a valid JSON file")

    return stac


def _load_stac_from_local(uri: str) -> dict[str, Any]:
    if not os.path.exists(uri):
        raise Exception(f"Could not locate file for the URI provided: {uri}")

    with open(uri) as file:
        try:
            stac = json.load(file)
        except json.decoder.JSONDecodeError:
            raise Exception("The provided URI does not point to a valid JSON file")

        return stac


def load_stac_json(uri: str):
    if type(uri) is not str:
        raise ValueError("Type of URI parameter must be a string.")

    if uri.startswith("http://") or uri.startswith("https://"):
        # uri is an url that points to a STAC
        stac = _load_stac_from_remote(uri)
    else:
        # assume uri points to a local file
        stac = _load_stac_from_local(uri)

    return stac
