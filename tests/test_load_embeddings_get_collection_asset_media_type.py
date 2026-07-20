import datetime

import pystac
import pytest
from pystac.extensions.item_assets import AssetDefinition

# Adjust this import to wherever the function lives
from openeo_processes_dask_ml.process_implementations.load_embeddings import (
    _get_collection_asset_media_type,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def make_collection(item_assets: dict | None = None) -> pystac.Collection:
    """Build a minimal valid pystac.Collection, optionally with item_assets."""
    collection = pystac.Collection(
        id="test-collection",
        description="A test collection",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=pystac.TemporalExtent(
                [[datetime.datetime(2020, 1, 1), datetime.datetime(2021, 1, 1)]]
            ),
        ),
    )
    if item_assets is not None:
        # item_assets maps asset_name -> AssetDefinition
        collection.item_assets = {
            name: AssetDefinition({"type": media_type})
            for name, media_type in item_assets.items()
        }
    return collection


def make_item(item_id: str, assets: dict) -> pystac.Item:
    """Build a minimal pystac.Item with the given {asset_name: media_type} assets."""
    item = pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
        },
        bbox=[0, 0, 1, 1],
        datetime=datetime.datetime(2020, 6, 1),
        properties={},
    )
    for name, media_type in assets.items():
        item.add_asset(
            name,
            pystac.Asset(
                href=f"https://example.com/{item_id}/{name}", media_type=media_type
            ),
        )
    return item


def make_item_collection(items: list[pystac.Item]) -> pystac.ItemCollection:
    return pystac.ItemCollection(items)


# --------------------------------------------------------------------------- #
# Tests: item_assets present on collection
# --------------------------------------------------------------------------- #
def test_returns_media_type_from_collection_item_assets():
    collection = make_collection(
        item_assets={"data": "image/tiff; application=geotiff"}
    )
    items = make_item_collection(
        [make_item("item1", {"data": "something/else"})]  # should be ignored
    )

    result = _get_collection_asset_media_type(collection, items, "data")

    assert result == "image/tiff; application=geotiff"


def test_item_assets_takes_precedence_over_items():
    # Even if items disagree, item_assets should win and no exception is raised.
    collection = make_collection(item_assets={"data": "image/png"})
    items = make_item_collection(
        [
            make_item("item1", {"data": "image/tiff"}),
            make_item("item2", {"data": "image/jpeg"}),
        ]
    )

    result = _get_collection_asset_media_type(collection, items, "data")

    assert result == "image/png"


# --------------------------------------------------------------------------- #
# Tests: no item_assets -> fall back to items
# --------------------------------------------------------------------------- #
def test_returns_media_type_from_items_when_no_item_assets():
    collection = make_collection(item_assets=None)
    items = make_item_collection(
        [
            make_item("item1", {"data": "image/tiff"}),
            make_item("item2", {"data": "image/tiff"}),
        ]
    )

    result = _get_collection_asset_media_type(collection, items, "data")

    assert result == "image/tiff"


def test_single_item_media_type():
    collection = make_collection(item_assets=None)
    items = make_item_collection([make_item("item1", {"data": "image/tiff"})])

    result = _get_collection_asset_media_type(collection, items, "data")

    assert result == "image/tiff"


def test_raises_when_items_have_different_media_types():
    collection = make_collection(item_assets=None)
    items = make_item_collection(
        [
            make_item("item1", {"data": "image/tiff"}),
            make_item("item2", {"data": "image/jpeg"}),
        ]
    )

    with pytest.raises(Exception):
        _get_collection_asset_media_type(collection, items, "data")


def test_raises_when_one_of_many_items_differs():
    collection = make_collection(item_assets=None)
    items = make_item_collection(
        [
            make_item("item1", {"data": "image/tiff"}),
            make_item("item2", {"data": "image/tiff"}),
            make_item("item3", {"data": "image/png"}),  # the odd one out
        ]
    )

    with pytest.raises(Exception):
        _get_collection_asset_media_type(collection, items, "data")


def test_ignores_other_assets_in_items():
    collection = make_collection(item_assets=None)
    items = make_item_collection(
        [
            make_item("item1", {"data": "image/tiff", "thumbnail": "image/png"}),
            make_item("item2", {"data": "image/tiff", "thumbnail": "image/jpeg"}),
        ]
    )

    # 'thumbnail' differs, but we only ask about 'data', which is consistent.
    result = _get_collection_asset_media_type(collection, items, "data")

    assert result == "image/tiff"


# --------------------------------------------------------------------------- #
# Tests: parametrized media types
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "media_type",
    [
        "image/tiff; application=geotiff",
        "image/png",
        "application/json",
        "application/x-netcdf",
    ],
)
def test_various_media_types_from_item_assets(media_type):
    collection = make_collection(item_assets={"data": media_type})
    items = make_item_collection([make_item("item1", {"data": media_type})])

    assert _get_collection_asset_media_type(collection, items, "data") == media_type
