import json
import os
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from stac_validator import stac_validator


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


def load_stac_json(uri: str) -> dict[str, Any]:
    if type(uri) is not str:
        raise ValueError("Type of URI parameter must be a string.")

    if uri.startswith("http://") or uri.startswith("https://"):
        # uri is an url that points to a STAC
        stac = _load_stac_from_remote(uri)
    else:
        # assume uri points to a local file
        stac = _load_stac_from_local(uri)

    return stac


# Method copied from
# https://github.com/Open-EO/openeo-processes-dask/blob/main/openeo_processes_dask/process_implementations/cubes/load.py
def _validate_stac(url):
    # todo: when emeddings extension is released, make core=False
    stac = stac_validator.StacValidate(url, core=True)
    is_valid_stac = stac.run()
    if not is_valid_stac:
        raise Exception(
            f"The provided link is not a valid STAC. stac-validator message: {stac.message}"
        )
    if len(stac.message) == 1:
        try:
            asset_type = stac.message[0]["asset_type"]
        except:
            raise Exception(f"stac-validator returned an error: {stac.message}")
    else:
        raise Exception(
            f"stac-validator returned multiple items, not supported yet. {stac.message}"
        )
    return asset_type


# Method copied from
# https://github.com/Open-EO/openeo-processes-dask/blob/main/openeo_processes_dask/process_implementations/cubes/load.py
def search_for_parent_catalog(url: str) -> tuple[str, str]:
    parsed_url = urlparse(url)
    root_url = parsed_url.scheme + "://" + parsed_url.netloc
    catalog_url = root_url
    url_parts = PurePosixPath(unquote(parsed_url.path)).parts
    collection_id = url_parts[-1]
    for p in url_parts:
        if p != "/":
            catalog_url = catalog_url + "/" + p
        try:
            asset_type = _validate_stac(catalog_url)
        except Exception as e:
            # logger.debug(e)
            continue
        if asset_type == "CATALOG":
            break
    if asset_type != "CATALOG":
        raise Exception(
            "It was not possible to find the root STAC Catalog starting from the provided Collection."
        )
    return catalog_url, collection_id
