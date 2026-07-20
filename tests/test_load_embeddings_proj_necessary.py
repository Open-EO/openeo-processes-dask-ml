import json
from datetime import datetime, timezone
from unittest import mock

import pystac
import pytest

# --- ADJUST THIS IMPORT to point at your module -----------------------------
from openeo_processes_dask_ml.process_implementations.load_embeddings import (
    _proj_necessary,
)

# ---------------------------------------------------------------------------

PROJ_EXT_URI = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
ASSET_NAME = "data"


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------
def make_collection(collection_proj_code=None, summary_proj_code=None):
    """Build a pystac.Collection, optionally with proj:code set on the
    collection itself and/or in its summaries."""
    collection = pystac.Collection(
        id="test-collection",
        description="test collection",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent(
                [[datetime(2020, 1, 1, tzinfo=timezone.utc), None]]
            ),
        ),
    )

    if collection_proj_code is not None:
        collection.stac_extensions.append(PROJ_EXT_URI)
        collection.extra_fields["proj:code"] = collection_proj_code

    if summary_proj_code is not None:
        if PROJ_EXT_URI not in collection.stac_extensions:
            collection.stac_extensions.append(PROJ_EXT_URI)
        # summaries store lists of allowed values
        collection.summaries.add("proj:code", list(summary_proj_code))

    return collection


def make_item(item_id, proj_code=None, asset_href="s3://bucket/file.parquet"):
    """Build a pystac.Item, optionally with a proj:code property and a
    geoparquet asset under ASSET_NAME."""
    item = pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
        },
        bbox=[0, 0, 1, 1],
        datetime=datetime(2021, 1, 1, tzinfo=timezone.utc),
        properties={},
    )

    if proj_code is not None:
        item.stac_extensions.append(PROJ_EXT_URI)
        item.properties["proj:code"] = proj_code

    if asset_href is not None:
        item.add_asset(
            ASSET_NAME,
            pystac.Asset(
                href=asset_href,
                media_type="application/x-parquet",
                roles=["data"],
            ),
        )
    return item


def make_item_collection(items):
    return pystac.ItemCollection(items=items)


# --- geoparquet metadata mocking -------------------------------------------
def geo_metadata(crs):
    """Return a dict shaped like the geoparquet 'geo' metadata block."""
    return {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": crs}},
    }


class FakeParquetMetadata:
    """Mimics the object returned by pyarrow.parquet.read_metadata()."""

    def __init__(self, crs, is_geoparquet=True):
        if is_geoparquet:
            self.metadata = {b"geo": json.dumps(geo_metadata(crs)).encode("utf-8")}
        else:
            # A regular (non-geo) parquet file: no 'geo' key.
            self.metadata = {b"pandas": b"{}"}


def patch_read_metadata(href_to_crs, non_geo_hrefs=()):
    """Return a mock side_effect for pyarrow.parquet.read_metadata that maps
    an href to a FakeParquetMetadata."""

    def _side_effect(href, *args, **kwargs):
        if href in non_geo_hrefs:
            return FakeParquetMetadata(None, is_geoparquet=False)
        return FakeParquetMetadata(href_to_crs[href], is_geoparquet=True)

    return _side_effect


# ---------------------------------------------------------------------------
# Step 1: proj:code on the collection or its summaries
# ---------------------------------------------------------------------------
class TestStep1CollectionLevel:
    def test_collection_has_proj_code_returns_false(self):
        collection = make_collection(collection_proj_code="EPSG:4326")
        items = make_item_collection(
            [make_item("a"), make_item("b")]  # no item-level proj:code
        )
        assert _proj_necessary(collection, items, ASSET_NAME) is False

    def test_summary_has_proj_code_returns_false(self):
        collection = make_collection(summary_proj_code=["EPSG:4326"])
        items = make_item_collection([make_item("a"), make_item("b")])
        assert _proj_necessary(collection, items, ASSET_NAME) is False

    def test_step1_short_circuits_and_does_not_open_files(self):
        """If step 1 resolves, we must never touch pyarrow."""
        collection = make_collection(collection_proj_code="EPSG:4326")
        items = make_item_collection([make_item("a"), make_item("b")])

        with mock.patch("pyarrow.parquet.read_schema") as read_schema:
            result = _proj_necessary(collection, items, ASSET_NAME)

        assert result is False
        read_schema.assert_not_called()


# ---------------------------------------------------------------------------
# Step 2: proj:code on every item
# ---------------------------------------------------------------------------
class TestStep2ItemLevel:
    def test_all_items_same_proj_code_returns_false(self):
        collection = make_collection()  # nothing at collection level
        items = make_item_collection(
            [
                make_item("a", proj_code="EPSG:4326"),
                make_item("b", proj_code="EPSG:4326"),
                make_item("c", proj_code="EPSG:4326"),
            ]
        )
        assert _proj_necessary(collection, items, ASSET_NAME) is False

    def test_items_different_proj_code_returns_true(self):
        collection = make_collection()
        items = make_item_collection(
            [
                make_item("a", proj_code="EPSG:4326"),
                make_item("b", proj_code="EPSG:3857"),
            ]
        )
        assert _proj_necessary(collection, items, ASSET_NAME) is True

    def test_step2_short_circuits_and_does_not_open_files(self):
        collection = make_collection()
        items = make_item_collection(
            [
                make_item("a", proj_code="EPSG:4326"),
                make_item("b", proj_code="EPSG:3857"),
            ]
        )
        with mock.patch("pyarrow.parquet.read_schema") as read_schema:
            result = _proj_necessary(collection, items, ASSET_NAME)

        assert result is True
        read_schema.assert_not_called()


# ---------------------------------------------------------------------------
# Step 3: fall back to reading the geoparquet CRS
# ---------------------------------------------------------------------------
class TestStep3GeoparquetCRS:
    def test_all_geoparquet_same_crs_returns_false(self):
        collection = make_collection()
        items = make_item_collection(
            [
                make_item("a", asset_href="s3://b/a.parquet"),
                make_item("b", asset_href="s3://b/b.parquet"),
            ]
        )
        crs = {"type": "name", "properties": {"name": "EPSG:4326"}}
        href_to_crs = {
            "s3://b/a.parquet": crs,
            "s3://b/b.parquet": crs,
        }
        with mock.patch(
            "pyarrow.parquet.read_schema",
            side_effect=patch_read_metadata(href_to_crs),
        ):
            result = _proj_necessary(collection, items, ASSET_NAME)

        assert result is False

    def test_geoparquet_different_crs_returns_true(self):
        collection = make_collection()
        items = make_item_collection(
            [
                make_item("a", asset_href="s3://b/a.parquet"),
                make_item("b", asset_href="s3://b/b.parquet"),
            ]
        )
        href_to_crs = {
            "s3://b/a.parquet": {
                "type": "name",
                "properties": {"name": "EPSG:4326"},
            },
            "s3://b/b.parquet": {
                "type": "name",
                "properties": {"name": "EPSG:3857"},
            },
        }
        with mock.patch(
            "pyarrow.parquet.read_schema",
            side_effect=patch_read_metadata(href_to_crs),
        ):
            result = _proj_necessary(collection, items, ASSET_NAME)

        assert result is True

    def test_non_geoparquet_raises_exception(self):
        collection = make_collection()
        items = make_item_collection(
            [
                make_item("a", asset_href="s3://b/a.parquet"),
                make_item("b", asset_href="s3://b/plain.parquet"),
            ]
        )
        href_to_crs = {
            "s3://b/a.parquet": {
                "type": "name",
                "properties": {"name": "EPSG:4326"},
            }
        }
        with mock.patch(
            "pyarrow.parquet.read_schema",
            side_effect=patch_read_metadata(
                href_to_crs, non_geo_hrefs=("s3://b/plain.parquet",)
            ),
        ):
            with pytest.raises(Exception, match="[Nn]on-?[Gg]eoparquet"):
                _proj_necessary(collection, items, ASSET_NAME)


# ---------------------------------------------------------------------------
# Mixed / edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_item_geoparquet_returns_false(self):
        """A single item can never have 'differing' CRS."""
        collection = make_collection()
        items = make_item_collection([make_item("a", asset_href="s3://b/a.parquet")])
        href_to_crs = {
            "s3://b/a.parquet": {
                "type": "name",
                "properties": {"name": "EPSG:4326"},
            }
        }
        with mock.patch(
            "pyarrow.parquet.read_schema",
            side_effect=patch_read_metadata(href_to_crs),
        ):
            assert _proj_necessary(collection, items, ASSET_NAME) is False

    def test_partial_item_proj_code_falls_through_to_step3(self):
        """If not *every* item has proj:code, step 2 doesn't resolve and we
        fall through to reading the files (step 3)."""
        collection = make_collection()
        items = make_item_collection(
            [
                make_item("a", proj_code="EPSG:4326", asset_href="s3://b/a.parquet"),
                make_item("b", asset_href="s3://b/b.parquet"),  # no proj:code
            ]
        )
        crs = {"type": "name", "properties": {"name": "EPSG:4326"}}
        href_to_crs = {
            "s3://b/a.parquet": crs,
            "s3://b/b.parquet": crs,
        }
        with mock.patch(
            "pyarrow.parquet.read_schema",
            side_effect=patch_read_metadata(href_to_crs),
        ) as read_metadata:
            result = _proj_necessary(collection, items, ASSET_NAME)

        assert result is False
        # confirms we actually went to step 3
        assert read_metadata.called
