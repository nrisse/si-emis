"""
This script compares KT-19 surface temperatures with ERA-5 and ESA sea ice 
temperature product.
"""

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from lizard.ac3airlib import day_of_flight
from lizard.readers.bbr import read_bbr
from lizard.readers.era5 import read_era5_single_levels
from lizard.readers.footprint import read_footprint
from scipy.stats import gaussian_kde

from si_emis.data_preparation.radiance import IR_EMISSIVITY
from si_emis.data_preparation.satellite import surftemp2sat
from si_emis.readers.ne23 import read_esa_cmems
from si_emis.retrieval.pamsim import select_times
from si_emis.retrieval.makesetups import times
from lizard.writers.figure_to_file import write_figure


def main():
    """
    Compares surface temperature from aircraft and different estimates on a
    map and along flight track
    """

    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    for flight_id in flight_ids:
        print(flight_id)

        along_track(flight_id)
        on_map(flight_id)


def along_track(flight_id):
    """
    Comparison along the flight track of the sea ice emissivity segments.
    """

    ds_fpr = get_kt19(flight_id)

    # get nearest ERA-5 and ESA pixel to every footprint
    da_era5 = surftemp2sat(
        ds_fpr, model="era-5", lon_var="lon", lat_var="lat", time_var="time"
    )
    da_esa = surftemp2sat(
        ds_fpr, model="esa", lon_var="lon", lat_var="lat", time_var="time"
    )

    dct_mod = {
        "ERA-5": da_era5,
        "CMEMS": da_esa,
    }

    # compare the three surface temperatures in a scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    for i, model in enumerate(dct_mod.keys()):
        x = ds_fpr.ts.copy()
        y = dct_mod[model]

        print(x.mean().item()-273.15)

        # drop positions where model is nan (e.g. over land for esa)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        # point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        ix = np.argsort(z)

        axes[i].scatter(x[ix], y[ix], c=z[ix], cmap="magma", s=1, lw=0)

        # data aspect
        axes[i].set_aspect("equal")

        # figure aspect
        axes[i].set_box_aspect(1)

        axes[i].axline(
            (265, 265), slope=1, color="k", linewidth=0.5, alpha=0.5
        )

        # compute and annotate errors
        rmse = np.round(calculate_rmse(x, y).item(), 1)
        bias = np.round(calculate_bias(x, y).item(), 1)
        mae = np.round(calculate_mae(x, y).item(), 1)
        mse = np.round(calculate_mse(x, y).item(), 1)
        axes[i].annotate(
            f"RMSE: {rmse} K\nBias: {bias} K\nMAE: {mae} K\nMSE: "
            f"{mse} K$^2$",
            xy=(0, 1),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )

        axes[i].set_xlabel(f"KT-19 Ts (e(IR)={IR_EMISSIVITY}) [K]")
        axes[i].set_ylabel(f"{model} Ts [K]")

    write_figure(fig, f"surftemp_comparison_{flight_id}.png")


def calculate_rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def calculate_bias(x, y):
    return np.mean(x - y)


def calculate_mae(x, y):
    return np.mean(np.abs(x - y))


def calculate_mse(x, y):
    return np.mean((x - y) ** 2)


def get_kt19(flight_id):
    """
    Reads KT-19 data and footprint dataset and reduces it to clear-sky segments

    Parameters
    ----------
    flight_id

    Returns
    -------
    ds_fpr: footprint dataset with KT-19 at nadir.
    """

    # read kt-19 data
    da_ts = read_bbr(flight_id).KT19

    # read footprint
    ds_fpr = read_footprint(flight_id)

    # combine kt-19 and footprint
    ds_fpr["ts"] = da_ts / IR_EMISSIVITY

    # reduce to clear-sky segments used for emissivity calculation
    ds_fpr = select_times(ds_fpr, times=times[flight_id])
    ds_fpr = ds_fpr.sel(view_ang=0).reset_coords(drop=True)

    return ds_fpr


def on_map(flight_id):
    """
    Visualize the different surface temperature estimates on a map.
    """

    ds_fpr = get_kt19(flight_id)

    roi = [
        ds_fpr["lon"].min().item() - 1,
        ds_fpr["lon"].max().item() + 1,
        ds_fpr["lat"].min().item() - 1,
        ds_fpr["lat"].max().item() + 1,
    ]

    ds_era5 = read_era5_single_levels(day_of_flight(flight_id), roi=roi)
    ds_era5 = ds_era5.isel(time=12)
    ds_esa = read_esa_cmems(day_of_flight(flight_id), roi=roi)

    bounds = np.arange(int(ds_fpr.ts.min().item()) - 5, 276, 1)
    cmap = plt.cm.jet.copy()
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 6),
        subplot_kw=dict(
            projection=ccrs.NorthPolarStereo(central_longitude=10)
        ),
        constrained_layout=True,
    )

    for ax in axes:
        ax.coastlines()
        ax.set_extent(roi)

        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=["left", "bottom"],
            xlocs=mticker.FixedLocator(np.arange(-180, 180, 10)),
            ylocs=mticker.FixedLocator(np.arange(50, 90, 1)),
            x_inline=False,
            y_inline=False,
            rotate_labels=False,
            linewidth=0.25,
            color="#1F7298",
            alpha=0.5,
        )

        ax.scatter(
            ds_fpr.lon,
            ds_fpr.lat,
            c=ds_fpr.ts,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            zorder=2,
        )

    # era-5
    im = axes[0].pcolormesh(
        ds_era5.longitude,
        ds_era5.latitude,
        ds_era5.skt,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        zorder=1,
    )
    fig.colorbar(im, ax=axes[0], label="ERA-5 Skin temperature [K]")

    # esa
    im = axes[1].pcolormesh(
        ds_esa.lon,
        ds_esa.lat,
        ds_esa.analysed_st,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        zorder=1,
    )
    fig.colorbar(
        im, ax=axes[1], label="ESA sea and ice surface temperature " "[K]"
    )

    write_figure(fig, f"surftemp_map_{flight_id}.png")

    plt.close()


if __name__ == "__main__":
    main()
