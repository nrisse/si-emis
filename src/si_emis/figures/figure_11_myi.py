"""
Visualizes multiyear ice concentration during AFLUX (note this data is not 
available in summer during the two ACLOUD flights)
"""

import string

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmcrameri.cm as cmc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpl_pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from lizard.ac3airlib import day_of_flight
from lizard.readers.amsr2_sic import read_amsr2_sic
from lizard.readers.multiyearice import read_myi
from lizard.writers.figure_to_file import write_figure

import si_emis.style
from si_emis.data_preparation.airsat import airborne_filter
from si_emis.figures.figure_03_case import prepare_data
from si_emis.figures.figure_07_satmap import add_scale
from si_emis.readers.emissivity import read_emissivity_aircraft


def main():
    """
    Plot MYI concentration for AFLUX flights.
    """

    flight_ids = [
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    # make plot
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7, 3.5),
        subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=10)},
        constrained_layout=True,
    )

    cmap = cmc.berlin
    norm = mcolors.BoundaryNorm(np.arange(0, 101, 10), cmap.N)
    ticks = np.arange(0, 101, 20)

    for i, flight_id in enumerate(flight_ids):
        axes[i].annotate(
            f"({string.ascii_lowercase[i]})",
            xy=(0, 1),
            ha="left",
            va="bottom",
            xycoords="axes fraction",
        )

        mission, platform, name = flight_id.split("_")
        axes[i].annotate(
            f"{mission} {name}",
            xy=(0.5, 1),
            ha="center",
            va="bottom",
            xycoords="axes fraction",
        )

        if i == 0:
            draw_labels = ["left", "bottom"]
        else:
            draw_labels = ["bottom"]

        # add land and coastline
        land_10m = cfeature.NaturalEarthFeature(
            category="physical",
            name="land",
            scale="10m",
            edgecolor="#7AB9DC",
            linewidth=0.5,
            facecolor="#CAE9FB",
        )
        axes[i].add_feature(land_10m)

        axes[i].gridlines(
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

        axes[i].set_extent([2, 15, 79.5, 81.5])

        add_scale(
            ax=axes[i],
            proj=ccrs.NorthPolarStereo(central_longitude=10),
            width=80,
            step=20,
            annotation_params=dict(
                color="k",
                bbox=dict(
                    boxstyle="square,pad=0",
                    ec="None",
                    fc="white",
                    alpha=0.7,
                    zorder=11,
                ),
            ),
        )

        # read data
        ds_sic, ds_myi, ds_ea = read_data(flight_id)

        # plot MYI
        im = axes[i].pcolormesh(
            ds_myi.lon,
            ds_myi.lat,
            ds_myi.MYI,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        # plot sea ice concentration 15 and 90 % as contour
        sic_grid = ccrs.NorthPolarStereo(
            central_longitude=10
        ).transform_points(
            ccrs.PlateCarree(), ds_sic.lon.values, ds_sic.lat.values
        )

        cont = axes[i].contour(
            sic_grid[:, :, 0],
            sic_grid[:, :, 1],
            ds_sic.sic,
            levels=[15],
            linestyles=["solid"],
            colors="k",
            linewidths=0.75,
        )

        # line with path effects
        axes[i].plot(
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

        # highlight case study for AFLUX RF08
        if i == 0:
            ds_case = prepare_data()
            axes[i].plot(
                ds_case.lon.sel(channel=8),
                ds_case.lat.sel(channel=8),
                color="gray",
                path_effects=[
                    mpl_pe.Stroke(
                        linewidth=1.25,
                        foreground="k",
                    ),
                    mpl_pe.Normal(),
                ],
                lw=1,
                transform=ccrs.PlateCarree(),
                label="RF08 case study",
            )

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        label="Multiyear ice concentration [%]",
        ticks=ticks,
        orientation="horizontal",
        shrink=0.5,
    )

    # for legend
    axes[0].plot([], [], color="k", label="15 % sea ice\nconcentration")

    # add legend in lower left corner of figure and make it not affect other axes
    leg = axes[0].legend(
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.15, -0.05),
        bbox_transform=fig.transFigure,
    )
    leg.set_in_layout(False)

    write_figure(fig, "fig11_myi.png")


def read_data(flight_id):
    """
    Read MYI and emissivity data for AFLUX flights.
    """

    mission, platform, name = flight_id.split("_")

    path = (
        f"/data/obs/campaigns/{mission.lower()}/auxiliary/sea_ice/daily_grid/"
    )

    ds_sic = read_amsr2_sic(day_of_flight(flight_id), path)
    ds_myi = read_myi(day_of_flight(flight_id))
    ds_ea = read_emissivity_aircraft(flight_id)

    # filter emissivity
    ds_ea = airborne_filter(ds_ea, drop_times=True)

    return ds_sic, ds_myi, ds_ea


if __name__ == "__main__":
    main()
