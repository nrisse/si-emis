"""
Assess spatial variability of sea ice emissivity on different scales.
"""

import os
from string import ascii_lowercase

import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from lizard.writers.figure_to_file import write_figure
from scipy.spatial import cKDTree

from si_emis.data_preparation.airsat import airborne_filter
from si_emis.figures.figure_03_case import open_imagery, plot_imagery
from si_emis.figures.figure_05_kmeans import apply_kmeans
from si_emis.figures.figure_07_satmap import add_scale, diameter_to_scattersize
from si_emis.figures.figure_08_airsat import satellite_index
from si_emis.readers.airsat import read_airsat
from si_emis.readers.emissivity import read_emissivity_aircraft
from si_emis.style import kmeans_colors, mirac_cols

FLIGHT_ID = "AFLUX_P5_RF08"

CRS_GEOG = ccrs.PlateCarree()
CRS_PROJ = ccrs.epsg("32631")  # this should match the sentinel-2 image


def main():
    """
    Creates plot with map on the left side and line plot on the right side.
    The map shows footprints from MiRAC, MHS, and AMSR2 at 89 GHz as well as
    MiRAC's 243 GHz channel and the clusters with sentinel image as background.
    The line plot shows the emissivity variability as a function of footprint
    size.

    Both plots are independent of each other, but here they are combined to
    one figure.
    """

    # make subgridspec with smaller plot on left
    fig = plt.figure(figsize=(7, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.1)

    # plot map on left
    ax1 = fig.add_subplot(gs[0, 0], projection=CRS_PROJ)
    ax2 = fig.add_subplot(gs[0, 1])

    # annotate axis letters
    for i, ax in enumerate([ax1, ax2]):
        ax.annotate(
            f"({ascii_lowercase[i]})",
            xy=(-0.02, 1),
            xycoords="axes fraction",
            ha="right",
            va="top",
        )

    print("Create line plot")
    main_line(fig, ax2)

    print("Create map")
    main_map(fig, ax1)

    write_figure(fig, "spatial_variability.png")


def main_map(fig, ax):
    """
    Plot map

    Parameters
    ----------
    gs : matplotlib.gridspec.GridSpec
        Gridspec object for the figure
    """

    # read satellite footprints with aircraft resampled to satellite
    ds_es = read_airsat(flight_id=FLIGHT_ID)

    # read aircraft emissivity at high resolution
    ds_ea = read_emissivity_aircraft(flight_id=FLIGHT_ID)
    ds_ea = airborne_filter(ds_ea, drop_times=True)

    # calculate sea ice emissivity clusters
    ds_km = apply_kmeans()

    # read sentinel image
    sentinel_file = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_emissivity/",
        "sentinel/AFLUX_RF08_20190331/rgb_shifted2_crop.tiff",
    )

    # select specific amsr2 overflight
    ix = satellite_index(
        ds_es, instrument="AMSR2", channel=13, angle0=0, angle1=90
    )
    ds_amsr2h = ds_es.sel(ix_sat_int=ix)

    # select granule that is closest to MiRAC and only a-scan pixels
    ds_amsr2h = ds_amsr2h.sel(ix_sat_int=(ds_amsr2h.granule == "036536"))

    # select specific mhs overflight
    ix = satellite_index(
        ds_es, instrument="MHS", channel=1, angle0=0, angle1=30
    )
    ds_mhs = ds_es.sel(ix_sat_int=ix)

    # select granule that is closest to MiRAC and only a-scan pixels
    ds_mhs = ds_mhs.sel(
        ix_sat_int=(ds_mhs.granule == "033894")
        & (ds_mhs.satellite == "Metop-B")
    )

    # select footprints that have center roughly within map boundaries
    ds_amsr2h = ds_amsr2h.sel(
        ix_sat_int=(ds_amsr2h.lat_sat > 80.65) & (ds_amsr2h.lat_sat < 80.85)
    )
    ds_mhs = ds_mhs.sel(
        ix_sat_int=(ds_mhs.lat_sat > 80.65) & (ds_mhs.lat_sat < 80.85)
    )

    # scane time of satellite
    print(ds_amsr2h.scan_time.mean().values)
    print(ds_mhs.scan_time.mean().values)

    print(ds_amsr2h.incidence_angle.mean().values)
    print(ds_mhs.incidence_angle.mean().values)

    # read sentinel image and get projection
    ds_landsat, ds_landsat_data, projection, ds_landsat_gt = open_imagery(
        sentinel_file
    )

    assert projection == CRS_PROJ

    # colorscale for emissivity
    de = 0.025
    cmap1 = cmc.batlow
    bnds1 = np.arange(0.6, 1.001, de)
    norm1 = mcolors.BoundaryNorm(bnds1, ncolors=cmap1.N)

    # plot map
    # ax.set_extent([4.68, 5.28, 80.63, 80.87])
    ax.set_extent([4.78, 5.18, 80.68, 80.82])

    # add gridlines
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=["left", "bottom"],
        xlocs=mticker.FixedLocator(np.arange(-50, 50, 0.2)),
        ylocs=mticker.FixedLocator(np.arange(75, 83, 0.1)),
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
        linewidth=0.25,
        color="k",
        alpha=0.5,
        zorder=6,
    )

    # plot sentinel image on map
    plot_imagery(ax, ds_landsat, ds_landsat_data, ds_landsat_gt)

    # fix state of image to keep scattersize constant
    fig.canvas.draw()

    # plot satellite footprints
    # amsr2 horizontal
    scatter_satellite(ax, ds_amsr2h, cmap1, norm1, diameter=5000)
    scatter_satellite(ax, ds_mhs, cmap1, norm1, diameter=15000)

    # plot aircraft footprints at 89 GHz
    channel = 1
    coords = projection.transform_points(
        CRS_GEOG,
        ds_ea.lon.sel(channel=channel),
        ds_ea.lat.sel(channel=channel),
    )
    im = ax.scatter(
        coords[:, 0] - 100,
        coords[:, 1],
        c=ds_ea.e.sel(channel=channel, surf_refl="L"),
        transform=projection,
        s=diameter_to_scattersize(ax, diameter=100),
        cmap=cmap1,
        norm=norm1,
        edgecolors="none",
    )
    fig.colorbar(im, ax=ax, label="Emissivity")

    # plot aircraft footprint at 243 GHz
    # channel = 8
    # coords = projection.transform_points(
    #    CRS_GEOG,
    #    ds_ea.lon.sel(channel=channel),
    #    ds_ea.lat.sel(channel=channel),
    # )
    # im = ax.scatter(
    #    coords[:, 0],
    #    coords[:, 1],
    #    c=ds_ea.e.sel(channel=channel, surf_refl="L"),
    #    transform=projection,
    #    s=diameter_to_scattersize(ax, diameter=100),
    #    cmap=cmap1,
    #    norm=norm1,
    #    edgecolors="none",
    # )

    # plot aircraft kmeans cluster slightly shifted to the side
    coords = projection.transform_points(
        CRS_GEOG, ds_km.lon.sel(channel=1), ds_km.lat.sel(channel=1)
    )
    ax.scatter(
        coords[:, 0] + 100,
        coords[:, 1],
        c=[kmeans_colors[i] for i in ds_km.km_label.values],
        transform=projection,
        s=diameter_to_scattersize(ax, diameter=100),
        edgecolors="none",
    )

    # annotate satellite label
    ax.annotate(
        "AMSR2",
        xy=(0.2, 0.7),
        xycoords="axes fraction",
        ha="center",
        va="center",
        color=cmap1(norm1(0.91)),
    )

    ax.annotate(
        "MHS",
        xy=(0.13, 0.92),
        xycoords="axes fraction",
        ha="center",
        va="center",
        color=cmap1(norm1(0.83)),
    )

    ax.annotate(
        "MiRAC",
        xy=(0.35, 0.55),
        xycoords="axes fraction",
        ha="center",
        va="center",
        color=cmap1(norm1(0.96)),
    )

    ax.annotate(
        "Cluster",
        xy=(0.61, 0.11),
        xycoords="axes fraction",
        ha="left",
        va="center",
        color=cmap1(norm1(0.6)),
    )

    # add scale
    add_scale(ax, width=2, step=0.5, proj=projection, annotation_params=dict())

    # add legend for clusters on the top left outside of the axis
    ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                label=i,
                markerfacecolor=color,
                markersize=10,
                linewidth=0,
                markeredgecolor="none",
            )
            for i, color in kmeans_colors.items()
        ],
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0, 0.5),
        title="Cluster",
    )


def scatter_satellite(ax, ds, cmap, norm, diameter):
    """Plots satellite footprints"""
    sc = ax.scatter(
        ds.lon_sat,
        ds.lat_sat,
        c=ds.e.sel(surf_refl="L"),
        edgecolors=cmap(norm(ds.e.sel(surf_refl="L"))),
        transform=CRS_GEOG,
        s=diameter_to_scattersize(ax, diameter),
        cmap=cmap,
        norm=norm,
        lw=2,
    )
    sc.set_facecolor("none")


def main_line(fig, ax):
    """
    Assess spatial variablity of sea ice emissivity on different scales.

    As a test only for AFLUX RF08 for now
    """

    # run calculation for all five flights
    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    # variability from satellites
    df_sat = variability_satellite_mean(flight_ids)

    min_diameter = 0  # this results in the standard deviation of dataset
    max_diameter = 20  # this is the standard deviation of 30 km diameter
    step = 1
    diameters = np.arange(min_diameter, max_diameter + step, step)
    diameters[0] = 0.1  # airborne footprint size

    channels = [1, 7, 8, 9]

    # store results in array
    arr_std = np.full(
        (len(flight_ids), len(diameters), len(channels)), fill_value=np.nan
    )

    for i, flight_id in enumerate(flight_ids):
        print(flight_id)

        arr_std[i, :, :] = run(flight_id, channels, diameters)

    # calculate campaign-wide averages, min, max
    da_std = xr.DataArray(
        data=arr_std,
        dims=("flight_id", "diameter", "channel"),
        coords={
            "flight_id": flight_ids,
            "diameter": diameters,
            "channel": channels,
        },
    )

    da_std["campaign"] = (
        "flight_id",
        ["ACLOUD", "ACLOUD", "AFLUX", "AFLUX", "AFLUX"],
    )
    da_std = da_std.swap_dims({"flight_id": "campaign"})
    da_std_mean = da_std.groupby("campaign").mean(dim="campaign")
    da_std_min = da_std.groupby("campaign").min(dim="campaign")
    da_std_max = da_std.groupby("campaign").max(dim="campaign")
    da_std = da_std.swap_dims({"campaign": "flight_id"})

    # plot results
    plot_results(fig, ax, da_std, da_std_mean, da_std_min, da_std_max, df_sat)

    # print statistics by comparison small scale and large scale
    for campaign in da_std_mean.campaign.values:
        for channel in da_std_mean.channel.values:
            # skip if no values are available
            if np.isnan(
                da_std_mean.sel(campaign=campaign, channel=channel)
            ).all():
                continue

            d50m = (
                da_std_mean.sel(campaign=campaign, channel=channel)
                .sel(diameter=0.1, method="nearest")
                .item()
            )
            d5km = (
                da_std_mean.sel(campaign=campaign, channel=channel)
                .sel(diameter=5, method="nearest")
                .item()
            )
            d16km = (
                da_std_mean.sel(campaign=campaign, channel=channel)
                .sel(diameter=16, method="nearest")
                .item()
            )

            iratio5 = int(round((d5km / d50m - 1) * 100, 0))
            iratio16 = int(round((d16km / d50m - 1) * 100, 0))
            ratio5 = int(round((d50m / d5km - 1) * 100, 0))
            ratio16 = int(round((d50m / d16km - 1) * 100, 0))

            print(
                f"{campaign}, {channel}: "
                f"100 m scale: {d50m} "
                f"5 km scale: {d5km} "
                f"16 km scale: {d16km} "
                f"sat5km/mirac: {iratio5}% "
                f"sat16km/mirac: {iratio16}% "
                f"mirac/sat5km: {ratio5}% "
                f"mirac/sat16km: {ratio16}% "
            )


def run(flight_id, channels, diameters):
    """
    Run the calculation for a single flight.
    """

    # read sea ice emissivity
    ds_e = read_emissivity_aircraft(flight_id)
    ds_e = airborne_filter(ds_e, drop_times=True)

    # select surface reflection type
    ds_e = ds_e.sel(surf_refl="L")

    # apply airborne filter
    arr_std = spatial_variability(ds_e, channels=channels, diameters=diameters)

    return arr_std


def spatial_variability(ds_e, channels, diameters):
    """
    Assess spatial variability of sea ice emissivity on different scales.

    Parameters
    ----------
    ds_e : xarray.Dataset
        Dataset containing sea ice emissivity
    channels : list
        List of channels to be considered

    Returns
    -------
    arr_std : np.ndarray
        Array containing the standard deviation of the mean emissivity for
        different diameters and channels
    """

    arr_std = np.full((len(diameters), len(channels)), fill_value=np.nan)

    for i_channel, channel in enumerate(channels):
        print(channel)

        # drop all values that are nan here
        ds_e_ch = ds_e.copy(deep=True)
        ds_e_ch = ds_e_ch.sel(
            time=ds_e_ch.e.sel(channel=channel).notnull().values
        )

        # skip channel if no values are left
        if not ds_e_ch.time.size:
            continue

        # project points to polar stereographic coordinates
        coords = ccrs.epsg(3413).transform_points(
            src_crs=CRS_GEOG,
            x=ds_e_ch.lon.sel(channel=channel),
            y=ds_e_ch.lat.sel(channel=channel),
        )[:, :2]
        ds_e_ch["x"] = ("time", coords[:, 0])
        ds_e_ch["y"] = ("time", coords[:, 1])

        radii = diameters / 2

        tree = cKDTree(data=coords)

        # calculate distance between each point and all other points
        dist, ix = tree.query(
            coords, k=coords.shape[0], distance_upper_bound=radii[-1] * 1e3
        )

        # cut off second dimension if no point is within distance
        keep = (dist != np.inf).any(axis=0)
        dist = dist[:, keep]
        ix = ix[:, keep]

        for i_radius, radius in enumerate(radii):
            ix_ = ix.copy()

            # set flag for points that are not within distance
            exclude = dist > radius * 1e3
            ix_[exclude] = 0

            # get emissivity and ts values for all points
            emis_matrix = ds_e_ch.e.sel(channel=channel).values[ix_]
            ts_matrix = ds_e_ch.ts.sel(channel=channel).values[ix_]

            # flag values that are not within distance
            emis_matrix[exclude] = np.nan
            ts_matrix[exclude] = np.nan

            # calculate tb at the surface
            tb_matrix = emis_matrix * ts_matrix

            # average the tb and ts values for all points
            tb_mean = np.nanmean(tb_matrix, axis=1)
            ts_mean = np.nanmean(ts_matrix, axis=1)

            # calculate emissivity for all points
            emis_mean = tb_mean / ts_mean

            # calculate std of mean emissivity for all points
            # arr_std[i_radius, i_channel] = np.nanstd(emis_mean)

            # calculate interquartile range of emissivity for all points
            arr_std[i_radius, i_channel] = np.nanpercentile(
                emis_mean, 75
            ) - np.nanpercentile(emis_mean, 25)

    return arr_std


def plot_results(fig, ax, da_std, da_std_mean, da_std_min, da_std_max, df_sat):
    """
    Plot results.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    da_std : xarray.DataArray
        Array containing the standard deviation of the mean emissivity for
        different diameters and channels
    da_std_mean : xarray.DataArray
        Array containing the mean standard deviation of the mean emissivity for
        different diameters and channels
    da_std_min : xarray.DataArray
        Array containing the minimum standard deviation of the mean emissivity
        for different diameters and channels
    da_std_max : xarray.DataArray
        Array containing the maximum standard deviation of the mean emissivity
        for different diameters and channels
    df_sat : pandas.DataFrame
        Dataframe containing the mean variability for ACLOUD and AFLUX flights
        from satellites
    """

    labels = {1: "89 GHz", 7: "183 GHz", 8: "243 GHz", 9: "340 GHz"}

    # satellite campaign statistics
    df_sat_campaign_mean = df_sat.groupby("campaign").mean()
    df_sat_campaign_min = df_sat.groupby("campaign").min()
    df_sat_campaign_max = df_sat.groupby("campaign").max()
    markers = {"ACLOUD": "o", "AFLUX": "x"}
    linewidths = {
        "ACLOUD": 0.5,
        "AFLUX": 0.75,
    }
    linestyles = {
        "ACLOUD": "-",
        "AFLUX": "--",
    }

    # this representation shows line for each flight, but hard to see anything
    # for flight_id in da_std.flight_id.values:
    #    for channel in da_std.channel.values:
    #        # skip if no values are available
    #        if np.isnan(
    #            da_std.sel(flight_id=flight_id, channel=channel)
    #        ).all():
    #            continue

    #        campaign = flight_id.split("_")[0]

    #        ax.plot(
    #            da_std.diameter,
    #            da_std.sel(flight_id=flight_id, channel=channel),
    #            color=mirac_cols[channel],
    #            label=f"{labels[channel]}, {campaign}",
    #            linewidth=linewidths[campaign],
    #            linestyle=linestyles[campaign],
    #            zorder=1,
    #        )

    for campaign in da_std_mean.campaign.values:
        for channel in da_std_mean.channel.values:
            # skip if no values are available
            if np.isnan(
                da_std_mean.sel(campaign=campaign, channel=channel)
            ).all():
                continue

            ax.plot(
                da_std_mean.diameter,
                da_std_mean.sel(campaign=campaign, channel=channel),
                color=mirac_cols[channel],
                label=f"{labels[channel]}, {campaign}",
                linestyle=linestyles[campaign],
                zorder=1,
            )

            # fill between min and max value
            ax.fill_between(
                da_std_mean.diameter,
                da_std_min.sel(campaign=campaign, channel=channel),
                da_std_max.sel(campaign=campaign, channel=channel),
                color=mirac_cols[channel],
                alpha=0.2,
                zorder=0,
                linewidth=0,
            )

    # add points for satellite 89 GHz
    # ax.scatter(
    #    [5, 15],
    #    df_sat_campaign_mean.loc[campaign, ["AMSR2 89 GHz", "MHS 89 GHz"]],
    #    color=mirac_cols[1],
    #    marker=markers[campaign],
    #    label="AMSR2/MHS 89 GHz, "+campaign,
    #    zorder=2,
    # )

    # same for 150 GHz
    # ax.scatter(
    #    [15],
    #    df_sat_campaign_mean.loc[campaign, ["MHS 150 GHz"]],
    #    color=mirac_cols[7],
    #    marker=markers[campaign],
    #    label="MHS 150 GHz, "+campaign,
    #    zorder=2,
    # )

    ax.grid()

    ax.set_xlabel("Footprint size [km]")
    ax.set_ylabel("Emissivity interquartile range")

    ax.set_yticks(np.arange(0, 0.21, 0.05))
    ax.set_yticks(np.arange(0, 0.21, 0.01), minor=True)

    ax.set_xticks(np.arange(0, 21, 5))
    ax.set_xticks(np.arange(0, 21, 1), minor=True)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.18)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True)


def variability_satellite_mean(flight_ids):
    """
    Calculates mean variability for ACLOUD and AFLUX flights

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the mean variability for ACLOUD and AFLUX flights
    """

    df_lst = []
    for flight_id in flight_ids:
        df_lst.append(
            pd.DataFrame(
                data=variability_satellite(flight_id), index=[flight_id]
            )
        )

    df = pd.concat(df_lst, axis=0)

    df["campaign"] = ["ACLOUD" if "ACLOUD" in i else "AFLUX" for i in df.index]

    return df


def variability_satellite(flight_id):
    """
    This calculates the spatial variability of the satellite footprints for
    a specific research flight, satellite instrument, and satellite channel.

    The satellite dataset was spatially collocated with MiRAC observations so
    that the footprints can be compared with MiRAC observations

    Parameters
    ----------
    flight_id : str
        Flight ID

    Returns
    -------
    emis_std : dict
        Dictionary containing the standard deviation of the emissivity for
        different satellite instruments and channels
    """

    emis_std = {}

    # read satellite emissivity
    ds_es = read_airsat(flight_id)

    # query amsr2 89 GHz
    ix = satellite_index(
        ds_es, instrument="AMSR2", channel=13, angle0=0, angle1=90
    )
    if ix.sum() > 0:
        emis_std["AMSR2 89 GHz"] = (
            ds_es.e.sel(ix_sat_int=ix, surf_refl="L")
            .std(dim="ix_sat_int")
            .item()
        )

    # query mhs 89 GHz
    ix = satellite_index(
        ds_es, instrument="MHS", channel=1, angle0=0, angle1=30
    )
    if ix.sum() > 0:
        emis_std["MHS 89 GHz"] = (
            ds_es.e.sel(ix_sat_int=ix, surf_refl="L")
            .std(dim="ix_sat_int")
            .item()
        )

    # query mhs 150 GHz
    ix = satellite_index(
        ds_es, instrument="MHS", channel=2, angle0=0, angle1=30
    )
    if ix.sum() > 0:
        emis_std["MHS 150 GHz"] = (
            ds_es.e.sel(ix_sat_int=ix, surf_refl="L")
            .std(dim="ix_sat_int")
            .item()
        )

    return emis_std


if __name__ == "__main__":
    main()
