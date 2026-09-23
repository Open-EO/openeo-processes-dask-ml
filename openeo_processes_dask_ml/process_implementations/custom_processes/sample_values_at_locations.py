import geopandas as gpd
import rioxarray
import shapely
import xarray as xr
import xvec
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.utils import dim_utils


def _project_geometries(geoms: gpd.GeoDataFrame, to_crs) -> gpd.GeoDataFrame:
    if geoms.crs.equals(to_crs):
        return geoms
    return geoms.to_crs(to_crs)


def _attach_secondary_coordinats(
    vector_cube: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    ignore_columns: list[str] | None = None,
) -> xr.DataArray:
    if ignore_columns is None:
        ignore_columns = []
    if "geometry" not in ignore_columns:
        ignore_columns.append("geometry")

    # attach provided attributes as secondary coords to geometry
    vector_cube = vector_cube.assign_coords(
        **{
            col_name: ("geometry", gdf[col_name])
            for col_name in gdf
            if col_name not in ignore_columns
        }
    )
    return vector_cube


def _sample_from_raster_cube(
    raster_cube: xr.DataArray, gdf: gpd.GeoDataFrame
) -> xr.DataArray:
    x_coord = dim_utils.get_x_dim_name(raster_cube)  # raises DimensionMissing
    y_coord = dim_utils.get_y_dim_name(raster_cube)  # raises DimensionMissing

    # reproject points to data's CRS
    to_crs = raster_cube.rio.crs
    gdf = _project_geometries(gdf, to_crs)

    # Clip gdf to spatial bounds of data
    minx, miny, maxx, maxy = raster_cube.rio.bounds()
    extent = shapely.box(minx, miny, maxx, maxy)
    gdf_clipped = gdf.clip(extent)

    extracted: xr.DataArray = raster_cube.xvec.extract_points(
        gdf_clipped.geometry, x_coord, y_coord, name="geometry"
    )

    extracted = _attach_secondary_coordinats(extracted, gdf_clipped)

    return extracted


def _sample_from_vector_cube(
    vector_cube: xr.DataArray, gdf: gpd.GeoDataFrame
) -> xr.DataArray:
    geom_dim = dim_utils.get_geometry_dim_name(vector_cube)  # raises DimensionMissing

    to_crs = vector_cube.xvec.geom_coords_indexed[geom_dim].crs
    gdf = _project_geometries(gdf, to_crs)

    # 1. Extract geometries into standard GeoDataFrames
    # The polygons get an automatic integer index (0 to N-1) which we'll use for slicing
    gdf_poly = gpd.GeoDataFrame(
        geometry=vector_cube.coords[geom_dim].values, crs=to_crs
    )

    # 2. Perform a Spatial Join
    # This matches each point to the polygon that contains it.
    # `index_right` will hold the integer position of the matching polygon.
    joined = gdf.sjoin(gdf_poly, how="inner", predicate="within")

    # 3. Slice the DataCube using the matched integer positions
    # If multiple points fall in one polygon, the data is automatically duplicated.
    # Points that fall outside all polygons are dropped (due to how="inner").
    sampled_vec_cube = vector_cube.isel(geometry=joined["index_right"].values)

    # 5. Swap out the polygons for the exact Point geometries
    sampled_vec_cube = sampled_vec_cube.assign_coords(geometry=joined.geometry.values)

    # 6. Re-initialize the xvec GeometryIndex on the new points
    sampled_vec_cube = sampled_vec_cube.xvec.set_geom_indexes("geometry", crs=to_crs)

    sampled_vec_cube = _attach_secondary_coordinats(
        sampled_vec_cube, joined, ignore_columns=["index_right"]
    )

    return sampled_vec_cube


def sample_values_at_locations(data: xr.DataArray, geometries: dict) -> xr.DataArray:
    # load data
    if isinstance(geometries, dict):
        # type GeoJSON
        try:
            try:
                geom_crs = geometries["crs"]["properties"]["name"]
            except KeyError:
                geom_crs = "epsg:4326"
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=geom_crs)
        except:
            raise ValueError("Provided points object is not a valid GeoJSON object")
    else:
        # Vector DAtaCube: Could be DataArray, GeoDataFrame, dask-GeoDataFrame
        raise NotImplementedError(
            "Not Implemented, Provide the point geometries as a geojson."
        )

    # check that all are poitns
    if not all(isinstance(p, shapely.Point) for p in gdf.geometry):
        raise ValueError(
            "All provided geometries in `points` arguemnt must be Point geometries. One or multiple non-Point geometries were encountered"
        )

    # if this variable is still None later, something has gone wrong...
    extracted = None

    # Scenario 1: data is a raster Datacube
    try:
        extracted = _sample_from_raster_cube(data, gdf)
    except DimensionMissing:  # raised if x or y dim are msising
        pass

    # Scenario 2: data is a vector Datacube
    try:
        extracted = _sample_from_vector_cube(data, gdf)
    except DimensionMissing:  # raised if geometry dim is missing
        pass

    if extracted is None:
        raise ValueError(
            "Unsuppoted datacube was provided in `data` argument. Must be a raster datacube or vector datacube."
        )

    # todo: check if any point at all has extracted values

    return extracted
