import xarray as xr


def load_embeddings(
    url: str,
    spatial_extent: list[float],
    temporal_extent: list[float],
    asset_name: str = "embeddings",
):
    # ehck if url is json

    # if url ist STAC compliant

    # check if is item or collection

    # if item:
    # 1) check media type: if zarr: load with zarr, if geotiff: laod with rasterio (?), if gpq load as geoparquet

    # if collection: filter by bbox and temporal
    # checkl item-asset media type: if raster (e.g. zarr, geotiff) raise not implemented for now (?)
    # if geoparquet: load all of them, sort out spatial and temporal to form datacube
    # then cut (again) by bbox and temp

    # reproject into a harmonized grid (wgs84)
    # rename dimensions to harmonize everything: x,y,time,embedding

    pass
