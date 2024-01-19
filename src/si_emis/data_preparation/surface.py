"""
Calculates surface mask based on Norwegian Polar Institute datasets.
"""

import cartopy.crs as ccrs
import geopandas
import numpy as np
import xarray as xr
from lizard.ac3airlib import day_of_flight
from lizard.readers.amsr2_sic import read_amsr2_sic
from lizard.readers.land import read_glaciers, read_land
from scipy import spatial
from shapely.geometry import Point


def land_ocean_shore_mask(lon, lat, buffer=150):
    """
    Applies land sea mask to lat/lon footprint location.

    Parameters
    -------
    lon: footprint longitude
    lat: footprint latitude
    buffer: distance that defines the shore. Footprints within this distance
       from the coastlines are defined as shore.

    Returns
    -------
    land/sea/shore mask.
    """

    # read land shapefile
    gdf_land = read_land()

    # apply buffer
    gdf_land_incr = gdf_land.copy()
    gdf_land_shri = gdf_land.copy()

    gdf_land_incr["geometry"] = gdf_land.geometry.buffer(distance=buffer)
    gdf_land_shri["geometry"] = gdf_land.geometry.buffer(distance=-buffer)

    # convert footprint location arrays to geopandas dataframe
    gdf_fpr = geopandas.GeoDataFrame(
        crs="epsg:4326",
        geometry=list(map(lambda x, y: Point(x, y), lon.values, lat.values)),
    ).to_crs(gdf_land.crs)

    # apply spatial join between footprint and land masks (nan: point outside)
    gdf_fpr_in_land = gdf_fpr.sjoin(gdf_land, how="left")
    gdf_fpr_in_land_incr = gdf_fpr.sjoin(gdf_land_incr, how="left")
    gdf_fpr_in_land_shri = gdf_fpr.sjoin(gdf_land_shri, how="left")

    # positive buffer: points may intersect multiple polygons. keep only first
    gdf_fpr_in_land_incr = gdf_fpr_in_land_incr[
        ~gdf_fpr_in_land_incr.index.duplicated(keep="first")
    ]

    # create boolean masks for shore and land
    ix_land = ~gdf_fpr_in_land["index_right"].isnull()
    ix_ocean = gdf_fpr_in_land["index_right"].isnull()
    ix_shore = (
        ~gdf_fpr_in_land_incr["index_right"].isnull()
        & gdf_fpr_in_land_shri["index_right"].isnull()
    )

    # add land mask
    land_mask = xr.zeros_like(lon, dtype="uint8")
    land_mask[ix_land.values] = 1

    # ocean mask
    ocean_mask = xr.zeros_like(lon, dtype="uint8")
    ocean_mask[ix_ocean.values] = 1

    # shore mask
    shore_mask = xr.zeros_like(lon, dtype="uint8")
    shore_mask[ix_shore.values] = 1

    return land_mask, ocean_mask, shore_mask


def glacier_mask(lon, lat):
    """
    Applies glacier mask to lat/lon footprint location over terrain.

    Parameters
    -------
    lon: footprint longitude
    lat: footprint latitude

    Returns
    -------
    glacier mask.
    """

    # read glacier shapefile
    gdf_glacier = read_glaciers()

    # convert footprint location arrays to geopandas dataframe
    gdf_fpr = geopandas.GeoDataFrame(
        crs="epsg:4326", geometry=list(map(lambda x, y: Point(x, y), lon, lat))
    ).to_crs(gdf_glacier.crs)

    # apply spatial join between footprint and glacier masks (nan: point
    # outside)
    gdf_fpr_in_glacier = gdf_fpr.sjoin(gdf_glacier, how="left")

    # create boolean masks for glacier and mixed
    ix_glacier = ~gdf_fpr_in_glacier["index_right"].isnull()

    # add glacier mask (0: not glacier, 1: glacier)
    mask = xr.zeros_like(lon, dtype="uint8")
    mask[ix_glacier.values] = 1

    return mask


def sic_mask(lon, lat, flight_id):
    """
    Applies sea ice mask to lat/lon footprint location over terrain. Also
    returns the sea ice concentration

    Parameters
    -------
    lon: footprint longitude
    lat: footprint latitude

    Returns
    -------
    sea ice concentration and sea ice mask.
    """

    mission, platform, name = flight_id.split("_")

    # get flight date
    date = day_of_flight(flight_id).strftime("%Y%m%d")

    # read sea ice of date
    ds_sic = read_amsr2_sic(
        date,
        path=f"/data/obs/campaigns/{mission.lower()}/auxiliary/sea_ice/daily_grid/",
    )

    # convert to north polar stereographic projection
    proj_in = ccrs.PlateCarree()
    proj_out = ccrs.epsg("3413")

    m_xy = proj_out.transform_points(
        proj_in, ds_sic.lon.values, ds_sic.lat.values
    )[:, :, :2]
    mx = m_xy[:, :, 0].flatten()
    my = m_xy[:, :, 1].flatten()

    t_xy = proj_out.transform_points(proj_in, lon.values, lat.values)[:, :2]
    tx = t_xy[:, 0]
    ty = t_xy[:, 1]

    # KDtree from sea ice data
    tree = spatial.cKDTree(np.array([mx, my]).T, leafsize=20)
    dist, ix = tree.query(np.array([tx, ty]).T)

    assert dist.max() < (np.sqrt(6250**2 + 6250**2) / 2 + 100)

    # create sea ice data array
    da_sic = xr.DataArray(
        data=ds_sic.sic.values.flatten()[ix],
        coords={"time": lon.time},
        dims="time",
        name="sic",
    )
    da_sic.attrs = dict(
        standard_name="sea_ice_concentration",
        long_name="sea ice concentration",
        units="percent",
        description="Sea ice concentration at footprint location extracted"
        " from asi-AMSR2 n6250 v5.4 by University of Bremen",
    )

    # create sea ice mask (use 15% threshold)
    da_si_mask = xr.where(da_sic > 15, 1, 0)

    da_si_mask.attrs = dict(
        standard_name="sea_ice_mask",
        long_name="sea ice mask",
        description="Sea ice mask at footprint location extracted"
        " from asi-AMSR2 n6250 v5.4 by University of Bremen",
    )

    return da_sic, da_si_mask
