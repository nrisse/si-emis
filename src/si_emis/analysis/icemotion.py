"""
Check ice motion in study area
"""

import os
import string

import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.patheffects as mpl_pe

from lizard.ac3airlib import day_of_flight
from lizard.mpltools import style
from lizard.writers.figure_to_file import write_figure
from si_emis.data_preparation.airsat import airborne_filter
from si_emis.readers.emissivity import read_emissivity_aircraft


def main():
    """
    Plot ice motion maps for the flight days. This also
    calculates the direction and velocity from u and v components.
    """

    map_crs = ccrs.NorthPolarStereo(central_longitude=10)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7, 4),
        subplot_kw=dict(projection=map_crs),
    )

    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    for i, flight_id in enumerate(flight_ids):

        print(flight_id)

        axes.flat[i].annotate(
            f"({string.ascii_lowercase[i]})",
            xy=(0, 1),
            ha="left",
            va="bottom",
            xycoords="axes fraction",
        )

        axes.flat[i].annotate(
            flight_id.replace("_P5_", " "),
            xy=(0.5, 1),
            ha="center",
            va="bottom",
            xycoords="axes fraction",
        )

        time = pd.Timestamp(day_of_flight(flight_id))

        ds = read_ice_motion(year=time.year)

        axes.flat[i].set_extent([2, 20, 79, 82])

        # add land and coastline
        land_10m = cfeature.NaturalEarthFeature(
            category="physical",
            name="land",
            scale="10m",
            edgecolor="#7AB9DC",
            linewidth=0.5,
            facecolor="#CAE9FB",
        )
        axes.flat[i].add_feature(land_10m)

        if i == 0:
            draw_labels = ["left"]
        elif i == 3:
            draw_labels = ["left", "bottom"]
        elif i == 4:
            draw_labels = ["bottom"]
        else:
            draw_labels = False

        axes.flat[i].gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=draw_labels,
            xlocs=mticker.FixedLocator(np.arange(-30, 30, 5)),
            ylocs=mticker.FixedLocator(np.arange(70, 90, 0.5)),
            x_inline=False,
            y_inline=False,
            rotate_labels=False,
            linewidth=0.25,
            color="#1F7298",
            alpha=0.5,
        )

        x, y, z = map_crs.transform_points(
            src_crs=ccrs.PlateCarree(),
            x=ds.longitude.values.T,  # there seems to be a bug in the product
            y=ds.latitude.values,
        ).T

        cmap = cmc.batlow
        norm = mcolors.BoundaryNorm(boundaries=np.arange(0, 25, 2), ncolors=cmap.N)
        im = axes.flat[i].pcolormesh(
            x,
            y,
            np.sqrt(ds.u**2 + ds.v**2).sel(time=time)  # cm/s to km/d
            * 0.01  # cm to m
            * 0.001  # m to km
            * 60  # s to min
            * 60  # min to h
            * 24,  # h to d
            cmap=cmap,
            norm=norm,
            transform=map_crs,
        )

        # line with path effects
        ds_ea = read_emissivity_aircraft(flight_id)
        ds_ea = airborne_filter(ds_ea, drop_times=True)
        axes.flat[i].plot(
            ds_ea.lon.sel(channel=8),
            ds_ea.lat.sel(channel=8),
            color="lightgray",
            path_effects=[
                mpl_pe.Stroke(
                    linewidth=1.25,
                    foreground="darkgray",
                ),
                mpl_pe.Normal(),
            ],
            lw=1,
            transform=ccrs.PlateCarree(),
            label="Polar 5",
        )


    fig.colorbar(im, ax=axes, label="Drift speed [km/d]")
    axes[-1, -1].remove()

    leg = axes[1, 1].legend(
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1.2, 0.5),
    )
    leg.set_in_layout(False)

    write_figure(fig, f"ice_drift.png")


def read_ice_motion(year):
    """
    Read ice motion data from nsidc on 25 km grid for northern hemisphere

    Parameters
    ----------
    year : int
        Year of ice motion data
    """

    year = str(year)

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sat/icemotion_nsidc",
            f"icemotion_daily_nh_25km_{year}0101_{year}1231_v4.1.nc",
        )
    )

    ds["time"] = ds.time.astype("datetime64[ns]")

    return ds


if __name__ == "__main__":
    main()
