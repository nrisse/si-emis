"""
This module calculates the footprint of an airborne instrument with a given
viewing angle over terrain. A terrain model of Svalbard is required.
"""


import os
import tempfile

import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
from osgeo import gdal
from scipy.spatial import KDTree
from shapely.geometry import Point
from shapely.ops import unary_union

from si_emis.data_preparation.transformation import Range2Location


class FootprintTerrain(Range2Location):
    """
    Calculates the footprint location on terrain surfaces using a digital
    elevation model
    """

    def __init__(
        self,
        alpha,
        beta,
        heading,
        pitch,
        roll,
        lon,
        lat,
        h,
        rs=np.array([0, 0, 0]),
    ):
        """
        Parameters
        ----------
        alpha: instrument's azimuth angle relative to platform (time-fixed)
        beta: instrument's viewing angle relative to platform (time-fixed)
        heading: aircraft rotation around its z-axis (flight direction)
        pitch: aircraft rotation around its x-axis (wing to wing)
        roll: aircraft rotation around its y-axis (front to back)
        lon: aircraft longitude
        lat: aircraft latitude
        h: aircraft altitude
        rs: distance between instrument and gps sensor (default: zero)
        """

        self.dtm_geod_footprint = np.nan
        self.los_geod_footprint = np.nan
        self.range_footprint = np.nan

        super().__init__(alpha, beta, heading, pitch, roll, lon, lat, h, rs)

    def dtm(self):
        """
        Extracts DTM along flight track and transforms it to 1D array.
        """

        with tempfile.TemporaryDirectory(dir=os.environ["PATH_CACHE"]) as td:
            file_track = os.path.join(td, "track.shp")
            file_dtm = os.path.join(td, "dtm.tif")

            # buffer flight track over land and export as shapefile
            gdf_gps = gpd.GeoDataFrame(
                crs="epsg:4326",
                geometry=list(
                    map(lambda xx, yy: Point(xx, yy), self.lon, self.lat)
                ),
            ).to_crs(epsg="25833")
            gdf_gps["geometry"] = gdf_gps.geometry.buffer(distance=5000)
            gdf_gps = gpd.GeoSeries(unary_union(gdf_gps["geometry"]))

            gdf_gps.to_file(file_track, crs="EPSG:25833")

            # extract raster along track
            gdal.Warp(
                file_dtm,
                os.path.join(
                    os.environ["PATH_SEC"], "data/dtm/NP_S0_DTM20/S0_DTM20.tif"
                ),
                cutlineDSName=file_track,
                cropToCutline=True,
                format="GTiff",
            )

            dtm_raster = gdal.Open(file_dtm)

            (
                upper_left_x,
                x_size,
                x_rotation,
                upper_left_y,
                y_rotation,
                y_size,
            ) = dtm_raster.GetGeoTransform()

            # height values to 1-D array
            z = dtm_raster.ReadAsArray()
            dim_y, dim_x = z.shape
            dtm_valid = z > -1e3
            z = z[dtm_valid].flatten()

            del dtm_raster  # close file

        # coordinates to 1-D array
        x_coords = np.arange(dim_x) * x_size + upper_left_x + (x_size / 2)
        y_coords = np.arange(dim_y) * y_size + upper_left_y + (y_size / 2)

        ix_y, ix_x = np.where(dtm_valid)
        x = x_coords[ix_x]
        y = y_coords[ix_y]

        # to ecef
        dtm_geoc = ccrs.Geocentric().transform_points(
            ccrs.UTM(33),
            x,
            y,
            z,
        )

        # to geod
        dtm_geod = ccrs.Geodetic().transform_points(
            ccrs.UTM(33),
            x,
            y,
            z,
        )

        return dtm_geoc, dtm_geod

    @staticmethod
    def terrain_intersection(dtm, los):
        """
        Get first terrain intersection. Threshold is set equal to the DTM grid
        spacing

        Returns
        -------
        ix_dtm: index of closest dtm pixel of the flattened dtm array
        ix_gps: index of the closest line of sight bin
        distance_dtm: distance to dtm
        """

        # convert coordinates to utm 33
        dtm_utm = ccrs.Geodetic().transform_points(
            ccrs.UTM(33),
            dtm[:, 0],
            dtm[:, 1],
            dtm[:, 2],
        )

        los_utm = ccrs.Geodetic().transform_points(
            ccrs.UTM(33),
            los[:, :, 0],
            los[:, :, 1],
            los[:, :, 2],
        )

        # build KD-tree for horizontal distance
        tree = KDTree(dtm_utm[:, :2], leafsize=20)

        # calculate distances and get corresponding indices
        distance, indices1d = tree.query(los_utm[:, :, :2])

        # get indices of first intersection along line of sight
        below_surface = los_utm[:, :, 2] < dtm_utm[indices1d, 2]
        ix_los = np.argmax(below_surface, axis=1)
        ix_dtm = indices1d[range(indices1d.shape[0]), ix_los]
        distance_dtm = distance[range(indices1d.shape[0]), ix_los]

        return ix_dtm, ix_los, distance_dtm

    def get_footprint(self):
        """
        Find the footprint locations for an entire flight.
        """

        # calculate line of sight
        r = np.append(np.arange(0, 500, 5), np.arange(500, 4000, 25))
        self.transform_ranges(r)

        # get dtm along flight track
        dtm_geoc, dtm_geod = self.dtm()

        # get dtm coordinates at footprint
        ix_dtm, ix_los, distance_dtm = self.terrain_intersection(
            dtm=dtm_geoc, los=self.r_geoc
        )

        assert distance_dtm.max() < 20

        self.dtm_geod_footprint = dtm_geod[ix_dtm]
        self.los_geod_footprint = self.r_geod[
            range(self.r_geod.shape[0]), ix_los
        ]
        self.range_footprint = r[ix_los]
