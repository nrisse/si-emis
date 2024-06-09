"""
Plot the segments used to derive emissivity on a map for ACLOUD and AFLUX.
"""

import string

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as mpl_pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lizard.ac3airlib import get_all_flights
from lizard.readers.footprint import read_footprint
from lizard.readers.mean_sea_ice import read_mean_sea_ice
from matplotlib import ticker as mticker

from si_emis.data_preparation.airsat import airborne_filter
from si_emis.figures.figure_07_satmap import add_scale
from si_emis.readers.emissivity import read_emissivity_aircraft
from si_emis.style import flight_colors, missions
from si_emis.writers import write_figure

FLIGHT_IDS = [
    "ACLOUD_P5_RF23",
    "ACLOUD_P5_RF25",
    "AFLUX_P5_RF08",
    "AFLUX_P5_RF14",
    "AFLUX_P5_RF15",
]


def main():
    """
    Plot maps
    """

    # properties
    z_sic = 0
    z_land = 1
    z_track = 2
    z_grid = 3
    z_rf_label = 4

    # setup map
    extent = (-2, 27, 77, 82.5)
    central_lon = (extent[1] + extent[0]) / 2

    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6, 4),
        subplot_kw=dict(projection=proj),
        constrained_layout=True,
    )

    ax_dct = {"ACLOUD": axes[0], "AFLUX": axes[1]}
    draw_labels = [["left", "bottom"], ["bottom"]]

    for i, (mission, ax) in enumerate(ax_dct.items()):
        t0 = missions[mission]["start"]
        t1 = missions[mission]["end"]
        ax.annotate(
            f"({string.ascii_lowercase[i]})",
            xy=(0.02, 0.98),
            ha="left",
            va="top",
            xycoords="axes fraction",
        )
        label = (
            f"{mission} "
            f'({pd.Timestamp(t0).strftime("%d %B")} to '
            f'{pd.Timestamp(t1).strftime("%d %B %Y")})'
        )
        ax.annotate(
            label,
            xy=(0.5, 1),
            ha="center",
            va="bottom",
            xycoords="axes fraction",
        )

        # set map extent (adapt to data)
        ax.set_extent(extents=extent, crs=data_crs)

        # add land and coastline
        land_10m = cfeature.NaturalEarthFeature(
            category="physical",
            name="land",
            scale="10m",
            edgecolor="#7AB9DC",
            linewidth=0.5,
            facecolor="#CAE9FB",
        )
        ax.add_feature(land_10m, zorder=z_land)

        ax.gridlines(
            crs=data_crs,
            draw_labels=draw_labels[i],
            xlocs=mticker.FixedLocator(np.arange(-30, 30, 10)),
            ylocs=mticker.FixedLocator(np.arange(70, 90, 2)),
            x_inline=False,
            y_inline=False,
            rotate_labels=False,
            linewidth=0.25,
            color="#1F7298",
            alpha=0.5,
            zorder=z_grid,
        )

    # add 100 km scale bar in bottom left corner
    for ax in axes:
        add_scale(
            ax=ax,
            proj=proj,
            width=100,
            step=25,
            annotation_params=dict(
                color="white",
                bbox=dict(
                    boxstyle="square,pad=0",
                    ec="None",
                    fc="None",
                    zorder=9,
                ),
            ),
        )

    # plot mean sea ice concentration over campaign period as background
    im = None
    for mission, ax in ax_dct.items():
        # read mean sea ice concentration
        ds_sic = read_mean_sea_ice(mission)

        im = ax.pcolormesh(
            ds_sic.lon,
            ds_sic.lat,
            ds_sic.sic,
            cmap="Blues_r",
            vmin=0,
            vmax=100,
            transform=data_crs,
            zorder=z_sic,
            shading="nearest",
        )

    # plot all flights
    for mission, ax in ax_dct.items():
        flight_ids = get_all_flights(mission, "P5")
        for i, flight_id in enumerate(flight_ids):
            ds = read_footprint(flight_id)

            # line with path effects
            ax.plot(
                ds.lon,
                ds.lat,
                color="lightgray",
                path_effects=[
                    mpl_pe.Stroke(
                        linewidth=1,
                        foreground="darkgray",
                    ),
                    mpl_pe.Normal(),
                ],
                lw=1,
                zorder=z_track,
                transform=ccrs.PlateCarree(),
            )

            # dummy for legend
            if i == 0:
                ax.plot(
                    [],
                    [],
                    color="lightgray",
                    label="Polar 5",
                    lw=1,
                    path_effects=[
                        mpl_pe.Stroke(
                            linewidth=1,
                            foreground="darkgray",
                        ),
                        mpl_pe.Normal(),
                    ],
                )

    # plot flights with emissivity values
    for mission, ax in ax_dct.items():
        flight_ids = [f for f in FLIGHT_IDS if mission in f]
        for flight_id in flight_ids:
            ds = read_emissivity_aircraft(flight_id)
            ds = airborne_filter(ds, drop_times=True)

            # line with path effects
            ax.plot(
                ds.lon,
                ds.lat,
                color=flight_colors[flight_id],
                path_effects=[
                    mpl_pe.Stroke(linewidth=2, foreground="k"),
                    mpl_pe.Normal(),
                ],
                lw=1,
                zorder=z_track,
                transform=ccrs.PlateCarree(),
            )

            # dummy for legend
            ax.plot(
                [],
                [],
                color=flight_colors[flight_id],
                label=flight_id.split("_")[-1],
                lw=1,
                path_effects=[
                    mpl_pe.Stroke(linewidth=2, foreground="k"),
                    mpl_pe.Normal(),
                ],
            )

    # add legend to both axis
    for ax in axes:
        ax.legend(
            loc="lower right",
            frameon=True,
            bbox_to_anchor=(1.02, -0.02),
        )

    fig.colorbar(
        im,
        ax=axes,
        label="Sea ice concentration [%]",
        orientation="horizontal",
        shrink=0.4,
    )

    write_figure(fig, "flightmap.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
