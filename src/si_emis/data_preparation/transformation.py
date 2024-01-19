"""
Converts range coordinates to geographic coordinates for an instrument
located onboard of an aircraft.

The conversions are based on the description in Mech et al. (2019).
"""


import cartopy.crs as ccrs
import numpy as np


class Range2Location:
    """
    Converts range coordinates to geographic coordinates for an instrument
    located onboard of an aircraft.

    Convention of input parameters:
    - pitch: positive for nose above horizon (i.e. during ascents)
    - roll: positive for right wing down (i.e. right curve)
    - heading: 0=north, 90=east, 180/-180=south, -90=west
    - alpha: 0=forward, 90=right wing, 180=backward, 270=left wing
    - beta: 0=down, 90=horizontal, 180=up

    Every array has the dimension (time, range, spatial):
        time: temporal dimension
        range: dimension for the range bins
        spatial: the 3D coordinates (x, y, z) in any reference system

    Further references:
    - https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#
      From_ECEF_to_ENU
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
        Executes the transformation.

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

        self.alpha = alpha
        self.beta = beta
        self.heading = heading
        self.pitch = pitch
        self.roll = roll
        self.lon = lon
        self.lat = lat
        self.h = h
        self.rs = rs

        self.r_s = np.nan  # sensor coordinates
        self.r_p = np.nan  # platform coordinates
        self.r_c = np.nan  # local coordinates
        self.r_geoc = np.nan  # geocentric coordinates
        self.r_geod = np.nan  # geodetic coordinates (lat/lon)

    def transform_sp(self, alpha, beta, rs):
        """
        Transform from sensor to platform reference frame.

        Parameters
        ----------
        alpha: instrument's azimuth angle relative to platform (time-fixed)
        beta: instrument's viewing angle relative to platform (time-fixed)
        rs: distance between instrument and gps sensor (default: zero)
        """

        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)

        Rsp_alpha = np.array(
            [
                [
                    [
                        [np.cos(alpha), np.sin(alpha), 0],
                        [-np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1],
                    ]
                ]
            ]
        )

        Rsp_beta = np.array(
            [
                [
                    [
                        [1, 0, 0],
                        [0, np.sin(beta), np.cos(beta)],
                        [0, -np.cos(beta), np.sin(beta)],
                    ]
                ]
            ]
        )

        Rsp = np.einsum("ghij,ghjk->ghik", Rsp_alpha, Rsp_beta)
        Ssp = np.zeros((1, len(self.r_s), 3))
        Ssp[0, 0, :] = rs
        a = np.einsum("ghij,ghj->ghi", Rsp, self.r_s)
        self.r_p = a + Ssp

    def transform_pc(self, heading, pitch, roll):
        """
        Transform from platform to local coordinate system.

        Parameters
        ----------
        heading: aircraft rotation around its z-axis (flight direction)
        pitch: aircraft rotation around its x-axis (wing to wing)
        roll: aircraft rotation around its y-axis (front to back)
        """

        heading = np.deg2rad(heading)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

        zero = np.zeros_like(roll)
        one = np.ones_like(roll)

        Rpc_heading = np.array(
            [
                [
                    [np.cos(heading), np.sin(heading), zero],
                    [-np.sin(heading), np.cos(heading), zero],
                    [zero, zero, one],
                ]
            ]
        ).transpose((3, 0, 1, 2))

        Rsp_pitch = np.array(
            [
                [
                    [one, zero, zero],
                    [zero, np.cos(pitch), -np.sin(pitch)],
                    [zero, np.sin(pitch), np.cos(pitch)],
                ]
            ]
        ).transpose((3, 0, 1, 2))

        Rsp_roll = np.array(
            [
                [
                    [np.cos(roll), zero, np.sin(roll)],
                    [zero, one, zero],
                    [-np.sin(roll), zero, np.cos(roll)],
                ]
            ]
        ).transpose((3, 0, 1, 2))

        a = np.einsum("ghij,ghjk->ghik", Rsp_pitch, Rsp_roll)
        Rcp = np.einsum("ghij,ghjk->ghik", Rpc_heading, a)
        self.r_c = np.einsum("ghij,ghj->ghi", Rcp, self.r_p)

    def transform_cg(self, lon, lat, h):
        """
        Transform local tangent plane coordinates (east-north-up or ENU) to
        geodetic and ECEF coordinates.

        Parameters
        ----------
        lon: aircraft longitude
        lat: aircraft latitude
        h: aircraft altitude
        """

        Scg = ccrs.Geocentric().transform_points(
            ccrs.Geodetic(),
            lon,
            lat,
            h,
        )[:, np.newaxis]

        lam = np.deg2rad(lon)
        phi = np.deg2rad(lat)

        zero = np.zeros_like(lam)

        Rcg = np.array(
            [
                [
                    -np.sin(lam),
                    -np.sin(phi) * np.cos(lam),
                    np.cos(phi) * np.cos(lam),
                ],
                [
                    np.cos(lam),
                    -np.sin(phi) * np.sin(lam),
                    np.cos(phi) * np.sin(lam),
                ],
                [zero, np.cos(phi), np.sin(phi)],
            ]
        )[:, np.newaxis].transpose((3, 1, 0, 2))

        self.r_geoc = np.einsum("ghij,ghj->ghi", Rcg, self.r_c) + Scg

        self.r_geod = ccrs.Geodetic().transform_points(
            ccrs.Geocentric(),
            self.r_geoc[:, :, 0],
            self.r_geoc[:, :, 1],
            self.r_geoc[:, :, 2],
        )

    def prepare_range(self, r):
        """
        Inserts range bins into a (time, range, spatial) array.

        Parameters
        ----------
        r: range bins
        """

        self.r_s = np.zeros((1, len(r), 3))
        self.r_s[0, :, 1] = r

    def transform_ranges(self, r):
        """
        Transform an array of range bins to geographic coordinates.

        Parameters
        -------
        r: range in meters from the instrument (time-fixed)
        """

        # array bins in sensor coordinates
        self.prepare_range(r)

        # apply transformations
        self.transform_sp(alpha=self.alpha, beta=self.beta, rs=self.rs)

        self.transform_pc(
            heading=self.heading, pitch=self.pitch, roll=self.roll
        )

        self.transform_cg(lon=self.lon, lat=self.lat, h=self.h)


class FootprintLocation(Range2Location):
    """
    Finds footprint location in the local reference frame, i.e., by neglecting
    effects of the Earth's curvature and terrain.

    https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane
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
        Executes the transformation. Earth curvature is neglected. The
        curvature is about 7.8 cm per km. This introduces slight height
        offsets after transforming from local to geographic coordinates.
        Offsets above 5 m will produce an error. Offsets below this are set
        to 0 m.

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

        self.theta = np.nan

        super().__init__(alpha, beta, heading, pitch, roll, lon, lat, h, rs)

    def ray_sfc_intersection(self, h):
        """
        Calculates the intersection between the line of sight and the surface
        in local coordinate system.

        All vectors are of shape (time, range, spatial)
        """

        # define surface
        sfc_n = np.array([[[0, 0, 1]]])  # normal
        sfc_o = np.array([[np.zeros(len(h)), np.zeros(len(h)), -h]]).transpose(
            (2, 0, 1)
        )  # origin

        # ray direction
        los_dir = np.diff(self.r_c, axis=1)
        los_o = self.r_c[:, np.newaxis, 0, :]

        ndotu = np.einsum("ghi,ghi->gh", sfc_n, los_dir)[:, :, np.newaxis]
        w = los_o - sfc_o
        si = -np.einsum("ghi,ghi->gh", sfc_n, w)[:, :, np.newaxis] / ndotu
        s = w + si * los_dir + sfc_o

        # set intersection point as new range coordinate
        self.r_c = s

    def incidence_angle(self):
        """
        Calculate incidence angle in local coordinate system
        """

        # define surface normal
        sfc_n = np.array([[[0, 0, 1]]])

        # ray direction
        los_dir = -np.diff(self.r_c, axis=1)

        self.theta = np.rad2deg(
            np.arccos(np.einsum("ghi,ghi->g", sfc_n, los_dir))
        )

    def get_footprint(self):
        """
        Calculates exact footprint location, neglecting Earth's curvature.
        """

        # define the viewing direction in sensor reference frame
        r = np.array([0, 1])
        self.prepare_range(r=r)

        # apply transformations
        self.transform_sp(alpha=self.alpha, beta=self.beta, rs=self.rs)

        self.transform_pc(
            heading=self.heading, pitch=self.pitch, roll=self.roll
        )

        # calculate incidence angle
        self.incidence_angle()

        # get footprint location and replace viewing direction in range array
        self.ray_sfc_intersection(h=self.h)

        self.transform_cg(lon=self.lon, lat=self.lat, h=self.h)

        # remove height residual from neglecting curvature
        print(self.r_geod[:, 0, 2].max())
        assert self.r_geod[:, 0, 2].max() < 5
        self.r_geod[:, 0, 2] = 0
