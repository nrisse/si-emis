"""
Create histogram of emissivities and brightness temperatures during ACLOUD and
AFLUX over sea ice as simulated with PAMTRA.
"""

import string

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lizard.readers.band_pass import read_band_pass_combination
from xhistogram.xarray import histogram

from si_emis.data_preparation.airsat import airborne_emissivity_merged
from si_emis.figures.figure_05_kmeans import match_miracs
from si_emis.style import flight_colors, mirac_pol_text
from si_emis.writers import write_figure


def compute_histograms(ds, dct_fi, flight_ids, bins, name):
    """
    Computes histograms of given variable for all flights together and for each
    research flight individually for a list of flights.

    Parameters
    ----------
    ds: airborne emissivity dataset
    dct_fi: dictionary that maps integer index with flight_id
    flight_ids: compute histogram that include these flight ids
    bins: histogram bin edges
    name: variable on which histogram is calculated
    """

    # sub-select flight_ids
    ds = ds.sel(time=ds.flight_id.isin(flight_ids))

    # reduce dict to selected flights
    dct_fi = {k: v for k, v in dct_fi.items() if v in flight_ids}

    # calculate histogram over all flights
    ds_hist = histogram(ds[name], bins=bins, dim=["time"], density=False)

    # calculate histogram for each flight
    bins_fi = np.array([-1] + list(dct_fi)) + 0.5
    ds_hist_fi = histogram(
        ds[name],
        ds.i_flight_id,
        bins=[bins, bins_fi],
        dim=["time"],
        density=False,
    )
    ds_hist_fi["flight_id"] = ("i_flight_id_bin", list(dct_fi.values()))
    ds_hist_fi = ds_hist_fi.swap_dims({"i_flight_id_bin": "flight_id"})

    return ds_hist, ds_hist_fi


def main():
    """
    Reads all data in standardized way, computes and plots histogram.
    """

    # read band pass information
    ds_bp = read_band_pass_combination("MiRAC-A", "MiRAC-P")

    # read emissivity dataset for list of flights
    ds, dct_fi = airborne_emissivity_merged(
        flight_ids=[
            "ACLOUD_P5_RF23",
            "ACLOUD_P5_RF25",
            "AFLUX_P5_RF08",
            "AFLUX_P5_RF14",
            "AFLUX_P5_RF15",
        ],
        filter_kwargs=dict(drop_times=True, dtb_keep_tb=True),
    )

    # plot histograms when the same samples are used
    # reduce dataset to specific channels and surface reflection
    #ds = ds.sel(surf_refl="L")

    # match MiRAC-A and MiRAC-P spatially and temporally
    #ds = match_miracs(ds)

    # drop times when emissivity is nan in any channel
    #ds = ds.sel(time=~ds.e.sel(channel=[1, 8]).isnull().any("channel"))

    # add surf refl
    #ds["e"] = ds["e"].expand_dims("surf_refl")

    # calculate total flight distance
    flight_distance(ds)

    ac = ["ACLOUD_P5_RF23", "ACLOUD_P5_RF25"]
    af = ["AFLUX_P5_RF08", "AFLUX_P5_RF14", "AFLUX_P5_RF15"]

    # understand emissivity above 1 at 89 GHz
    above1_times = ds.e.sel(surf_refl="L", channel=1) > 1
    print(
        f"Number of 89 GHz Tb above 1 for Lambertian reflection:"
        f" {above1_times.sum().item()}"
    )
    ds.time.sel(time=above1_times)

    # understand lower-peaking 89 GHz emissivity at 89 GHz during ACLOUD
    ix = (ds.e.sel(surf_refl="L", channel=1) < 0.72) & (ds.i_flight_id == 0)
    ds.time.sel(time=ix)

    # compute emissivity histograms
    bins = np.arange(0, 1.11, 0.025)
    ds_hist_ac_e, ds_hist_fi_ac_e = compute_histograms(
        ds, dct_fi, flight_ids=ac, bins=bins, name="e"
    )
    ds_hist_af_e, ds_hist_fi_af_e = compute_histograms(
        ds, dct_fi, flight_ids=af, bins=bins, name="e"
    )

    # compute tb histograms
    bins = np.arange(140, 281, 5)
    ds_hist_ac_tb, ds_hist_fi_ac_tb = compute_histograms(
        ds, dct_fi, flight_ids=ac, bins=bins, name="tb"
    )
    ds_hist_af_tb, ds_hist_fi_af_tb = compute_histograms(
        ds, dct_fi, flight_ids=af, bins=bins, name="tb"
    )

    channels = [1, 7, 8, 9]

    # plot emissivity histogram
    kwds = dict(
        channels=channels,
        bin_var="e_bin",
        xlabel="Emissivity",
        fname="emissivity",
        xlim=[0.5, 1],
    )
    histograms_2x2(
        ds_bp,
        ds_hist_ac_e,
        ds_hist_af_e,
        ds_hist_fi_ac_e,
        ds_hist_fi_af_e,
        surf_refl="L",
        **kwds,
    )
    histograms_2x2(
        ds_bp,
        ds_hist_ac_e,
        ds_hist_af_e,
        ds_hist_fi_ac_e,
        ds_hist_fi_af_e,
        surf_refl="S",
        **kwds,
    )

    # plot tb histogram
    kwds = dict(
        channels=channels,
        bin_var="tb_bin",
        xlabel="TB [K]",
        fname="tb",
        xlim=[150, 280],
    )
    histograms_2x2(
        ds_bp,
        ds_hist_ac_tb,
        ds_hist_af_tb,
        ds_hist_fi_ac_tb,
        ds_hist_fi_af_tb,
        surf_refl="L",
        **kwds,
    )
    histograms_2x2(
        ds_bp,
        ds_hist_ac_tb,
        ds_hist_af_tb,
        ds_hist_fi_ac_tb,
        ds_hist_fi_af_tb,
        surf_refl="S",
        **kwds,
    )

    # plot emissivity and tb histograms together (Lambertian only)
    histograms_4x2(
        ds_bp,
        [ds_hist_ac_tb, ds_hist_ac_e],
        [ds_hist_af_tb, ds_hist_af_e],
        [ds_hist_fi_ac_tb, ds_hist_fi_ac_e],
        [ds_hist_fi_af_tb, ds_hist_fi_af_e],
        surf_refl="L",
        channels=channels,
        bin_var=["tb_bin", "e_bin"],
        xlabel=["TB [K]", "Emissivity"],
        fname="tb_emissivity_4x2",
        xlim=[[150, 280], [0.5, 1]],
    )

    histograms_2x4(
        ds_bp,
        [ds_hist_ac_tb, ds_hist_ac_e],
        [ds_hist_af_tb, ds_hist_af_e],
        [ds_hist_fi_ac_tb, ds_hist_fi_ac_e],
        [ds_hist_fi_af_tb, ds_hist_fi_af_e],
        surf_refl="L",
        channels=channels,
        bin_var=["tb_bin", "e_bin"],
        xlabel=["TB [K]", "Emissivity"],
        fname="tb_emissivity",
        xlim=[[150, 280], [0.5, 1]],
    )


def flight_distance(ds):
    """
    Calculate flight distance
    """

    import cartopy.crs as ccrs

    p = ccrs.epsg(3413).transform_points(
        src_crs=ccrs.PlateCarree(),
        x=ds.subac_lon.values,
        y=ds.subac_lat.values,
    )

    diff = p[1:, :2] - p[:-1, :2]

    ix = ds.time.diff("time") / np.timedelta64(1, "s") == 1

    total_dist = np.sum(np.sqrt(np.sum(diff[ix] ** 2, axis=1))) * 1e-3

    print(f"Total flight distance (EPSG=3413): {round(total_dist)} km")


def plot_histogram(
    axes,
    ds_hist_ac,
    ds_hist_af,
    ds_hist_fi_ac,
    ds_hist_fi_af,
    surf_refl,
    channels,
    bin_var,
):
    """
    Plots histogram of sea ice emissivity for all research flights.
    All histograms are normalized by the count in each campaign

    Parameters
    ----------
    axes: axes to plot on (flattened)
    ds_hist_ac: histogram during acloud
    ds_hist_af: histogram during aflux
    ds_hist_fi_ac: histogram for each flight during acloud
    ds_hist_fi_af: histogram for each flight during aflux
    surf_refl: surface reflection type
    channels: channels to be plotted (should be four currently)
    bin_var: bin variable name (e.g. e_bin)
    """

    if "surf_refl" in ds_hist_fi_ac.dims:
        kwds = dict(surf_refl=surf_refl)
    else:
        kwds = dict()

    # do not plot the data if there are less than 10 observations available
    ix = ds_hist_fi_ac.sel(**kwds).sum(bin_var) < 10
    ds_hist_fi_ac = xr.where(ix, 0, ds_hist_fi_ac)

    ix = ds_hist_fi_af.sel(**kwds).sum(bin_var) < 10
    ds_hist_fi_af = xr.where(ix, 0, ds_hist_fi_af)

    # assumes that bin width is constant
    width = ds_hist_fi_ac[bin_var].diff(bin_var)[0].item()

    # plot data
    for i, channel in enumerate(channels):
        n_ac = ds_hist_fi_ac.sel(**kwds, channel=channel).sum(
            (bin_var, "flight_id")
        )

        n_af = ds_hist_fi_af.sel(**kwds, channel=channel).sum(
            (bin_var, "flight_id")
        )

        # plot probability by dividing by the number of observations and the
        # bin width
        # all flights acloud
        axes[i].step(
            ds_hist_ac[bin_var],
            ds_hist_ac.sel(**kwds, channel=channel) / n_ac / width,
            where="mid",
            linewidth=1,
            color="darkgray",
            zorder=2,
            label="ACLOUD",
        )

        # all flights aflux
        axes[i].step(
            ds_hist_af[bin_var],
            ds_hist_af.sel(**kwds, channel=channel) / n_af / width,
            where="mid",
            linewidth=1,
            color="k",
            zorder=3,
            label="AFLUX",
        )

        # individual flights acloud
        bottom = xr.zeros_like(ds_hist_fi_ac[bin_var])
        for flight_id in ds_hist_fi_ac.flight_id.values:
            axes[i].bar(
                ds_hist_fi_ac[bin_var],
                ds_hist_fi_ac.sel(**kwds, channel=channel, flight_id=flight_id)
                / n_ac
                / width,
                width=width,
                label=flight_id.replace("_P5_", " "),
                bottom=bottom,
                align="center",
                linewidth=0,
                color=flight_colors[flight_id],
                zorder=0,
            )
            bottom += (
                ds_hist_fi_ac.sel(**kwds, channel=channel, flight_id=flight_id)
                / n_ac
                / width
            )

        # individual flights aflux
        bottom = xr.zeros_like(ds_hist_fi_af[bin_var])
        for flight_id in ds_hist_fi_af.flight_id.values:
            axes[i].bar(
                ds_hist_fi_af[bin_var],
                ds_hist_fi_af.sel(**kwds, channel=channel, flight_id=flight_id)
                / n_af
                / width,
                width=width,
                label=flight_id.replace("_P5_", " "),
                bottom=bottom,
                align="center",
                linewidth=0,
                color=flight_colors[flight_id],
                zorder=1,
                alpha=0.5,
            )
            bottom += (
                ds_hist_fi_af.sel(**kwds, channel=channel, flight_id=flight_id)
                / n_af
                / width
            )


def histograms_2x2(
    ds_bp,
    ds_hist_ac,
    ds_hist_af,
    ds_hist_fi_ac,
    ds_hist_fi_af,
    surf_refl,
    channels,
    bin_var,
    xlabel,
    fname,
    xlim,
):
    """
    Histograms in a 2x2 arrangement for either emissivity of Tb

    Parameters
    ----------
    ds_bp: bandpass dataset
    ds_hist_ac: histogram during acloud
    ds_hist_af: histogram during aflux
    ds_hist_fi_ac: histogram for each flight during acloud
    ds_hist_fi_af: histogram for each flight during aflux
    surf_refl: surface reflection type
    channels: channels to be plotted (should be four currently)
    bin_var: bin variable name (e.g. e_bin)
    xlabel: x axis label
    fname: file name that is used
    xlim: x axis limites
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(5, 5),
        sharex="all",
        sharey="all",
        constrained_layout=True,
    )

    # annotate channel names
    for i, channel in enumerate(channels):
        label = ds_bp.label.sel(channel=channel).item()
        fig.axes[i].annotate(
            f"{label} ({mirac_pol_text[channel]})",
            xy=(0.5, 1),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
        )

    # call plotting function and give it data and axis
    plot_histogram(
        axes.flatten(),
        ds_hist_ac,
        ds_hist_af,
        ds_hist_fi_ac,
        ds_hist_fi_af,
        surf_refl,
        channels,
        bin_var,
    )

    axes[0, 0].set_xlim(xlim)
    axes[0, 0].set_ylim(bottom=0)
    axes[-1, 0].set_xlabel(xlabel)
    axes[-1, 0].set_ylabel("Density [%]")

    axes[0, 0].legend(
        loc="upper left",
        frameon=False,
        fontsize=5.5,
        bbox_to_anchor=(-0.02, 1.02),
    )

    # annotate letters
    for i, ax in enumerate(fig.axes):
        txt = f"({string.ascii_lowercase[i]})"
        ax.annotate(
            txt, xy=(0, 1), xycoords="axes fraction", ha="right", va="bottom"
        )

    write_figure(fig, f"{fname}_histogram_{surf_refl}.png")

    plt.close()


def histograms_4x2(
    ds_bp,
    ds_hist_ac,
    ds_hist_af,
    ds_hist_fi_ac,
    ds_hist_fi_af,
    surf_refl,
    channels,
    bin_var,
    xlabel,
    fname,
    xlim,
):
    """
    Histograms of Tb and emissivity in a 2x4 arrangement. The Tb/Emissivity is
    provided in the rows and the columns are the four MiRAC frequencies.
    Parameters
    ----------
    ds_bp: bandpass dataset
    ds_hist_ac: histogram during acloud
    ds_hist_af: histogram during aflux
    ds_hist_fi_ac: histogram for each flight during acloud
    ds_hist_fi_af: histogram for each flight during aflux
    surf_refl: surface reflection type
    channels: channels to be plotted (should be four currently)
    bin_var: bin variable name (e.g. e_bin)
    xlabel: x axis label
    fname: file name that is used
    xlim: x axis limits
    """

    fig, axes = plt.subplots(4, 2, figsize=(4, 6), sharey="col", sharex="col")

    # annotate channel names
    for i, channel in enumerate(channels):
        label = ds_bp.label.sel(channel=channel).item()
        for j in range(2):
            axes[i, j].annotate(
                f"{label} ({mirac_pol_text[channel]})",
                xy=(0.5, 1),
                xycoords="axes fraction",
                ha="center",
                va="bottom",
            )

    # call plotting function for tb and emissivity one after another
    for i in range(2):
        plot_histogram(
            axes[:, i].flatten(),
            ds_hist_ac[i],
            ds_hist_af[i],
            ds_hist_fi_ac[i],
            ds_hist_fi_af[i],
            surf_refl,
            channels,
            bin_var[i],
        )

        axes[0, i].set_xlim(xlim[i])
        axes[0, i].set_ylim(bottom=0)
        axes[-1, i].set_xlabel(xlabel[i])

    axes[-1, 0].set_ylabel("Density [%]")

    axes[0, 0].legend(
        loc="upper left",
        frameon=False,
        fontsize=5.5,
        bbox_to_anchor=(-0.02, 1.02),
    )

    axes[0, 1].set_xticks(np.arange(0.6, 1.2, 0.2))

    # annotate letters
    for i, ax in enumerate(fig.axes):
        txt = f"({string.ascii_lowercase[i]})"
        ax.annotate(
            txt, xy=(0, 1), xycoords="axes fraction", ha="right", va="bottom"
        )

    write_figure(fig, f"{fname}_histogram_{surf_refl}.png")

    plt.close()


def histograms_2x4(
    ds_bp,
    ds_hist_ac,
    ds_hist_af,
    ds_hist_fi_ac,
    ds_hist_fi_af,
    surf_refl,
    channels,
    bin_var,
    xlabel,
    fname,
    xlim,
):
    """
    Histograms of Tb and emissivity in a 2x4 arrangement. The Tb/Emissivity is
    provided in the rows and the columns are the four MiRAC frequencies.
    Parameters
    ----------
    ds_bp: bandpass dataset
    ds_hist_ac: histogram during acloud
    ds_hist_af: histogram during aflux
    ds_hist_fi_ac: histogram for each flight during acloud
    ds_hist_fi_af: histogram for each flight during aflux
    surf_refl: surface reflection type
    channels: channels to be plotted (should be four currently)
    bin_var: bin variable name (e.g. e_bin)
    xlabel: x axis label
    fname: file name that is used
    xlim: x axis limits
    """

    fig, axes = plt.subplots(
        3, 4, figsize=(7, 4), sharey="row", sharex="row",
        gridspec_kw=dict(bottom=0.2),
        height_ratios=[1, 1, 0.5],
    )
    for ax in axes[2, :]:
        ax.axis("off")
    axes = axes[:2, :]

    # annotate channel names
    for i, channel in enumerate(channels):
        label = ds_bp.label.sel(channel=channel).item()

        if "$" in label:
            label = label.split("$")[0] + " GHz"

        axes[0, i].annotate(
            f"{label} ({mirac_pol_text[channel]})",
            xy=(0.5, 1),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
        )

    # call plotting function for tb and emissivity one after another
    for i in range(2):
        plot_histogram(
            axes[i, :].flatten(),
            ds_hist_ac[i],
            ds_hist_af[i],
            ds_hist_fi_ac[i],
            ds_hist_fi_af[i],
            surf_refl,
            channels,
            bin_var[i],
        )

        axes[i, 0].set_xlim(xlim[i])
        axes[i, 0].set_ylim(bottom=0)
        axes[i, 0].set_xlabel(xlabel[i])
        axes[i, 1].set_xlabel(xlabel[i])
        axes[i, 2].set_xlabel(xlabel[i])
        axes[i, 3].set_xlabel(xlabel[i])

    axes[0, 0].set_ylabel("Density [K$^{-1}$]")
    axes[-1, 0].set_ylabel("Density")

    leg = axes[0, 1].legend(
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
        ncol=4,
    )

    # place legend in bottom without making it affect the subplot
    leg.set_in_layout(False)

    axes[1, 0].set_xticks(np.arange(0.6, 1.001, 0.2))
    axes[1, 0].set_xticks(np.arange(0.5, 1.001, 0.025), minor=True)
    axes[0, 0].set_xticks(np.arange(150, 281, 5), minor=True)

    # annotate letters
    for i, ax in enumerate(axes.flatten()):
        txt = f"({string.ascii_lowercase[i]})"
        ax.annotate(
            txt, xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top"
        )

    write_figure(fig, f"{fname}_histogram_{surf_refl}.png")

    plt.close()


if __name__ == "__main__":
    main()
