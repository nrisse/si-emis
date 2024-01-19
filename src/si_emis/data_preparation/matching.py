"""
Contains routines to match footprint from nadir and off-nadir instruments.
Requires calculated footprint locations.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lizard.readers.footprint import read_footprint
from scipy.spatial import KDTree


class Matching:
    """
    Matches footprints from MiRAC-A and MiRAC-P instruments based on
    precalculated footprint positions

    Example:
    match = Matching('ACLOUD_P5_RF23', from_angle=0, to_angle=-25)
    match.match(threshold=200, temporally=True)
    match.evaluate_hist()
    match.evaluate_map()
    match.evaluate_time_height()
    """

    def __init__(self, ds_fpr, fpr_from, fpr_to):
        """
        Reads footprint dataset from flight_id. Both footprint dataset should
        have the same time dimension

        Example: KT-19 at 0 deg (from_angle) to MiRAC-A at -25 (to_angle)
        - for each MiRAC-A footprint, the matching KT-19 footprint is searched,
          if any footprint is closer than a threshold.
        - all MiRAC-A footprints are conserved, while some KT-19 footprints
          might get lost during the matching process.

        Parameters
        ----------
        ds_fpr: dataset that contains footprint and temporal information
        fpr_from: first footprint coordinates
        fpr_to: footprint coordinates to which first one is matched
        """

        self.ds_fpr = ds_fpr
        self.fpr_from = fpr_from
        self.fpr_to = fpr_to
        self.distance = np.nan

    @classmethod
    def from_flight_id(cls, flight_id, from_angle=0, to_angle=-25):
        """
        Reads footprint dataset from flight id.

        Parameters
        ----------
        flight_id: research flight id
        from_angle: initial instrument viewing angle
        to_angle: target instrument viewing angle
        """

        ds_fpr = read_footprint(flight_id)

        # calculate footprint positions in utm 33n
        fpr_from = ccrs.UTM(33).transform_points(
            ccrs.PlateCarree(),
            ds_fpr["lon"].sel(view_ang=from_angle),
            ds_fpr["lat"].sel(view_ang=from_angle),
            ds_fpr["surf_alt"].sel(view_ang=from_angle),
        )

        fpr_to = ccrs.UTM(33).transform_points(
            ccrs.PlateCarree(),
            ds_fpr["lon"].sel(view_ang=to_angle),
            ds_fpr["lat"].sel(view_ang=to_angle),
            ds_fpr["surf_alt"].sel(view_ang=to_angle),
        )

        return cls(ds_fpr, fpr_from, fpr_to)

    @classmethod
    def from_dataset(cls, ds, from_channel, to_channel):
        """
        Uses footprint from a provided dataset (e.g. emissivity dataset).

        Parameters
        ----------
        ds: dataset
        from_channel: initial channel with specific viewing angle
        to_channel: target channel with another viewing angle
        """

        # calculate footprint positions in utm 33n
        fpr_from = ccrs.UTM(33).transform_points(
            ccrs.PlateCarree(),
            ds["lon"].sel(channel=from_channel),
            ds["lat"].sel(channel=from_channel),
            ds["surf_alt"].sel(channel=from_channel),
        )

        fpr_to = ccrs.UTM(33).transform_points(
            ccrs.PlateCarree(),
            ds["lon"].sel(channel=to_channel),
            ds["lat"].sel(channel=to_channel),
            ds["surf_alt"].sel(channel=to_channel),
        )

        return cls(ds, fpr_from, fpr_to)

    def match(self, threshold=200, temporally=True):
        """
        Matches footprint of nadir and off-nadir instrument

        Parameters
        ----------
        threshold: distance threshold (footprints within this distance match)
        temporally: add time dimension as well (seconds are added as 4th dim)
        """

        if temporally:
            time_progress = (self.ds_fpr.time - self.ds_fpr.time[0]).values
            time_progress = (
                time_progress.astype("timedelta64[s]").astype("int")
                / (30 * 60)
                * 100
            )
            self.fpr_from = np.append(
                self.fpr_from, time_progress[:, np.newaxis], axis=1
            )
            self.fpr_to = np.append(
                self.fpr_to, time_progress[:, np.newaxis], axis=1
            )

        tree = KDTree(self.fpr_from)
        self.distance, ix = tree.query(self.fpr_to)
        ix[self.distance > threshold] = -1

        # this index has to be applied on the 'from' viewing angle
        self.ds_fpr["from_time"] = xr.DataArray(
            data=self.ds_fpr.time.isel(time=ix),
            coords={"time": self.ds_fpr.time},
        )

        # has to be applied after indexing values
        self.ds_fpr["from_valid"] = ("time", self.distance <= threshold)

    def evaluate_hist(self):
        """
        Distributions of distances.
        """

        fig, ax = plt.subplots(1, 1)

        ax.hist(
            self.distance,
            bins=np.arange(0, 1000, 1),
            cumulative=True,
            density=True,
        )

        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1)

        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("CDF")

        return fig, ax

    def evaluate_map(self):
        """
        Plot footprint on map with distance as color.

        Works only for viewing angle coordinate
        """

        # plot map with distance as color
        fig, ax = plt.subplots(
            1,
            1,
            subplot_kw=dict(
                projection=ccrs.NorthPolarStereo(central_longitude=10)
            ),
        )

        ax.coastlines()

        ax.scatter(
            self.ds_fpr.subac_lon,
            self.ds_fpr.subac_lat,
            c="#dddddd",
            transform=ccrs.PlateCarree(),
        )

        ax.scatter(
            self.fpr_from[:, 0],
            self.fpr_from[:, 1],
            c="#aaaaaa",
            transform=ccrs.UTM(33),
        )

        ax.scatter(
            self.fpr_to[:, 0],
            self.fpr_to[:, 1],
            c=self.distance,
            transform=ccrs.UTM(33),
        )

        ax.scatter(
            self.ds_fpr.lon.sel(view_ang=self.from_angle)
            .sel(time=self.ds_fpr["from_time"])
            .where(self.ds_fpr.from_valid.values),
            self.ds_fpr.lat.sel(view_ang=self.from_angle)
            .sel(time=self.ds_fpr["from_time"])
            .where(self.ds_fpr.from_valid.values),
            color="#777777",
            transform=ccrs.PlateCarree(),
        )

        # connect matching points
        ds_from = (
            self.ds_fpr.sel(view_ang=self.from_angle)
            .sel(time=self.ds_fpr["from_time"])
            .where(self.ds_fpr.from_valid.values)
        )

        ds_to = self.ds_fpr.sel(view_ang=self.to_angle).where(
            self.ds_fpr.from_valid.values
        )

        empty = xr.full_like(ds_to.lat, fill_value=np.nan)

        x = np.vstack(
            [ds_from.lon.values, ds_to.lon.values, empty.values]
        ).T.ravel()
        y = np.vstack(
            [ds_from.lat.values, ds_to.lat.values, empty.values]
        ).T.ravel()

        ax.plot(x, y, transform=ccrs.PlateCarree())

        return fig, ax

    def evaluate_time_height(self):
        """
        Plots time height diagram

        Works only for viewing angle coordinate
        """

        # plot time/height plot with matching footprints
        fig, ax = plt.subplots(1, 1)

        # altitudes
        ax.plot(
            self.ds_fpr.time,
            self.ds_fpr.ac_alt,
            marker=".",
            label="aircraft",
            color="k",
        )

        ax.plot(
            self.ds_fpr.time,
            self.ds_fpr.surf_alt.sel(view_ang=self.from_angle),
            marker=".",
            label="from",
            color="green",
        )

        ax.plot(
            self.ds_fpr.time,
            self.ds_fpr.surf_alt.sel(view_ang=self.to_angle),
            marker=".",
            label="to",
            color="gray",
        )

        ax.plot(
            self.ds_fpr.time,
            self.ds_fpr.surf_alt.sel(view_ang=self.from_angle)
            .sel(time=self.ds_fpr["from_time"])
            .where(self.ds_fpr.from_valid.values),
            marker="x",
            label="from->to",
            color="green",
        )

        ax.legend()
