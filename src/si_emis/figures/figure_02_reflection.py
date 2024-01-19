"""
Estimate the impact of surface reflection type on the derived emissivity and
brightness temperature at 183 GHz.
"""

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lizard.readers.band_pass import read_band_pass_combination
from lizard.writers.figure_to_file import write_figure
from xhistogram.xarray import histogram

from si_emis.figures.figure_04_hist import airborne_emissivity_merged
from si_emis.readers.emissivity import read_emissivity_aircraft
from si_emis.retrieval.emissivity import (
    prepare_pamtra,
    query_aircraft_setups,
    read_setups,
)
from si_emis.style import mirac_cols, sensors

xr.set_options(display_max_rows=30)

CH_CENTER = 6
CH_WING = 7

COLOR_S = cmc.batlow(0.25)
COLOR_L = cmc.batlow(0.75)


def main():
    """
    Identifies, which surface reflection type produces the best results at
    183 GHz.
    """

    flight_ids_aflux = ["AFLUX_P5_RF08", "AFLUX_P5_RF14", "AFLUX_P5_RF15"]
    flight_ids_acloud = ["ACLOUD_P5_RF23", "ACLOUD_P5_RF25"]

    flight_ids = flight_ids_aflux

    ds, dct_flights = airborne_emissivity_merged(
        flight_ids=flight_ids,
        filter_kwargs=dict(
            dtb_filter=False, drop_times=True, angle_filter=True
        ),
        without_offsets=False,
    )

    ds["tb_sim"] = tb_equation(
        e=ds.e.sel(channel=7).reset_coords(drop=True),
        tb_e0=ds.tb_e0,
        tb_e1=ds.tb_e1,
    )

    # calculate bias
    ds["tb_bias"] = ds.tb.sel(channel=CH_CENTER) - ds.tb_sim.sel(
        channel=CH_CENTER
    )

    # plot time series
    time_series(ds, i_offset=0)
    time_series(ds, i_offset=1)
    time_series(ds, i_offset=2)
    time_series(ds, i_offset=3)
    time_series(ds, i_offset=4)
    time_series(ds, i_offset=5)
    time_series(ds, i_offset=6)

    # plot histograms
    plot_histograms(ds)

    emissivity_difference(ds)

    tb_difference(ds)


def tb_difference(ds):
    """
    Calculated the Tb difference that occurs when using the derived Lambertian
    emissivity above a specular surface and vice versa. This provides Tb values
    to the emissivity difference to show their relevance.

    Theoretical estimates of this were done by Matzler et al. (2005)

    Parameters
    ----------
    ds: emissivity dataset from Polar 5
    """

    # calculate tb simulation with the Lambertian PAMTRA simulation and
    # derived specular emissivity
    ds["tb_assume_l"] = tb_equation(
        e=ds.e.sel(surf_refl="L").reset_coords(drop=True),
        tb_e0=ds.tb_e0.sel(surf_refl="S").reset_coords(drop=True),
        tb_e1=ds.tb_e1.sel(surf_refl="S").reset_coords(drop=True),
    )

    # same for Lambertian
    ds["tb_assume_s"] = tb_equation(
        e=ds.e.sel(surf_refl="S").reset_coords(drop=True),
        tb_e0=ds.tb_e0.sel(surf_refl="L").reset_coords(drop=True),
        tb_e1=ds.tb_e1.sel(surf_refl="L").reset_coords(drop=True),
    )

    # compute difference to the observed Tb, which is the one corresponding to
    # the "real" emissivity
    ds["tb_bias_assume_l"] = ds["tb"] - ds["tb_assume_l"]
    ds["tb_bias_assume_s"] = ds["tb"] - ds["tb_assume_s"]

    # plot the Tb bias as a function of Lambertian emissivity and compare with
    # emissivity difference. Does one see the frequencies? How much difference?
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for channel in [1, 7, 8, 9]:
        kwds = dict(i_offset=3, channel=channel)

        ax.scatter(
            ds.e.sel(surf_refl="S", **kwds) - ds.e.sel(surf_refl="L", **kwds),
            ds.tb_bias_assume_s.sel(**kwds),
            color=mirac_cols[channel],
        )

    ax.set_xlabel("$e_s$ - $e_L$")
    ax.set_ylabel("$Tb_{obs}$ - $Tb_{sim,L}(e_s)$")

    ax.set_ylim(top=0)
    ax.set_xlim(left=0)


def emissivity_difference(ds):
    """
    Difference between Lambertian and specular emissivity for every channel as
    a function of Lambertian emissivity. This shows, how much emissivity
    depends on the selection of the surface reflection type.

    Parameters
    ----------
    ds: emissivity dataset from Polar 5
    """

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for channel in [1, 7, 8, 9]:
        kwds = dict(i_offset=3, channel=channel)

        ax.scatter(
            ds.e.sel(surf_refl="L", **kwds),
            ds.e.sel(surf_refl="S", **kwds) - ds.e.sel(surf_refl="L", **kwds),
            color=mirac_cols[channel],
        )

    ax.set_xlabel("$e_L$")
    ax.set_ylabel("$e_s$ - $e_L$")

    ax.set_ylim(bottom=0)


def time_series(ds_tb, i_offset):
    """
    Plot time series to identify if some ice types are specular and others are
    not. This approach will be new and was not conducted in any previous study.
    It will help to understand reflection type preferences at 183 GHz.

    Color-coding: specular is coral and Lambertian is blue

    Returns
    -------

    """

    offset = dict(i_offset=i_offset)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True, figsize=(6, 7))

    # observed tb at 183+/-5 GHz
    ax0.scatter(
        ds_tb.time, ds_tb.tb.sel(channel=CH_CENTER), color="k", s=5, lw=0
    )

    # simulated tb based on emissivity at 183+/-7.5 GHz channel (Lambertian)
    ax0.scatter(
        ds_tb.time,
        ds_tb.tb_sim.sel(channel=CH_CENTER, surf_refl="L", **offset),
        color=COLOR_S,
        s=5,
        lw=0,
    )

    # simulated tb based on emissivity at 183+/-7.5 GHz channel (specular)
    ax0.scatter(
        ds_tb.time,
        ds_tb.tb_sim.sel(channel=CH_CENTER, surf_refl="S", **offset),
        color=COLOR_L,
        s=5,
        lw=0,
    )

    # difference between observation and simulation based on specular refl
    ax1.scatter(
        ds_tb.time,
        ds_tb.tb.sel(channel=CH_CENTER)
        - ds_tb.tb_sim.sel(channel=CH_CENTER, surf_refl="S", **offset),
        color=COLOR_S,
        s=5,
        lw=0,
    )

    # difference between observation and simulation based on Lambertien refl
    ax1.scatter(
        ds_tb.time,
        ds_tb.tb.sel(channel=CH_CENTER)
        - ds_tb.tb_sim.sel(channel=CH_CENTER, surf_refl="L", **offset),
        color=COLOR_L,
        s=5,
        lw=0,
    )

    # panel 3: emissivity
    # derived emissivity at 183+/-7.5 GHz under specular reflection
    ax2.scatter(
        ds_tb.time,
        ds_tb.e.sel(channel=CH_WING, surf_refl="S", **offset),
        color=COLOR_S,
        s=5,
        lw=0,
    )

    # derived emissivity at 183+/-7.5 GHz under Lambertian reflection
    ax2.scatter(
        ds_tb.time,
        ds_tb.e.sel(channel=CH_WING, surf_refl="L", **offset),
        color=COLOR_L,
        s=5,
        lw=0,
    )

    # panel 4: emissivity difference
    # difference of derived emissivity between 183+/-5 and +/-7.5 GHz
    # specular
    ax3.scatter(
        ds_tb.time,
        ds_tb.e.sel(channel=CH_CENTER, surf_refl="S", **offset)
        - ds_tb.e.sel(channel=CH_WING, surf_refl="S", **offset),
        color=COLOR_S,
        s=5,
        lw=0,
    )

    # lambertian
    ax3.scatter(
        ds_tb.time,
        ds_tb.e.sel(channel=CH_CENTER, surf_refl="L", **offset)
        - ds_tb.e.sel(channel=CH_WING, surf_refl="L", **offset),
        color=COLOR_L,
        s=5,
        lw=0,
    )

    # aid lines
    ax1.axhline(y=0, color="k", linewidth=0.75)
    ax3.axhline(y=0, color="k", linewidth=0.75)

    # labels
    ax0.set_ylabel("Tb [K]")
    ax1.set_ylabel(r"$\Delta$Tb [K]")
    ax2.set_ylabel("$e$")
    ax3.set_ylabel("$\Delta e$")
    ax3.set_xlabel("Time [UTC]")


def plot_histograms(ds):
    """
    Plot difference between forward simulation based on either specular or
    Lambertian emissivity and radiative transfer and observation. The
    emissivity was calculated at 183+/-7.5 GHz and applied to forward
    simulation of 183+/-5 GHz.

    Approach from Guedj et al. (2010), where it was applied to AMSU-A at
    50 GHz band for different specularity parameters. Specularity parameters
    are not tested here, because of relatively high uncertainty in the 183 GHz
    emissivity.

    Parameters
    ----------
    ds: emissivity dataset with the calculated bias at channel 6 (183+/-5 GHz)
    """

    bias_bins = np.arange(-10, 10.5, 0.5)

    da_hist = histogram(
        ds["tb_bias"], bins=bias_bins, dim=["time"], density=True
    )

    # to avoid lines visible at the bottom
    da_hist = da_hist.where(da_hist > 0, -1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    colors = {"L": COLOR_L, "S": COLOR_S}
    label = {"L": "Lambertian", "S": "specular"}
    yposition = {"L": 0.39, "S": 0.39}

    for surf_refl in ds.surf_refl.values:
        # bias distribution
        ax.step(
            da_hist.tb_bias_bin,
            da_hist.sel(surf_refl=surf_refl, i_offset=3),
            color=colors[surf_refl],
            label=label[surf_refl],
            where="mid",
        )

        ax.fill_between(
            x=da_hist.tb_bias_bin,
            y1=0,
            y2=da_hist.sel(surf_refl=surf_refl, i_offset=3),
            step="mid",
            color=colors[surf_refl],
            alpha=0.25,
        )

        # add errorbar around the mean bias with error being the maximum mean
        # bias per time step for the sensitivity tests
        bias = ds["tb_bias"].sel(surf_refl=surf_refl, i_offset=3).mean()
        bias_l = ds["tb_bias"].min("i_offset").sel(surf_refl=surf_refl).mean()
        bias_u = ds["tb_bias"].max("i_offset").sel(surf_refl=surf_refl).mean()

        print(
            f"{round(bias.item(), 1)} ({round(bias_l.item(), 1)}-"
            f"{round(bias_u.item(), 1)})"
        )

        ax.errorbar(
            x=bias,
            y=yposition[surf_refl],
            xerr=np.array([bias - bias_l, bias_u - bias])[:, np.newaxis],
            color=colors[surf_refl],
            marker="o",
            capsize=5,
            capthick=1,
            label=f"Mean$\pm$uncertainty\n({label[surf_refl]})",
        )

    ax.set_ylabel("Density [K$^{-1}$]")
    ax.set_xlabel("$T_{b,\mathrm{obs}}-T_{b,\mathrm{sim}}$ [K]")

    ax.axvline(x=0, color="k", linestyle=":")

    ax.legend(loc="upper left", frameon=False)

    ax.set_ylim(0, 0.4)
    ax.set_xlim(-8, 8)

    write_figure(fig, "reflection_183ghz.png")

    plt.close()


def angular_dependence():
    """
    Look at angular dependence of bias between observation and simulation.
    It would be expected that the bias decreases with increasing angle,
    because the specular downwelling part comes closer to the effective angle.
    """

    # simple scatter
    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        ds.ac_zen.sel(channel=CH_CENTER),
        ds.tb_bias.sel(surf_refl="L", i_offset=3),
        alpha=0.2,
    )

    ax.scatter(
        ds.ac_zen.sel(channel=CH_CENTER),
        ds.tb_bias.sel(surf_refl="S", i_offset=3),
        alpha=0.2,
    )

    ax.axhline(y=0)


def compute_tb(da_e, ds_tb, file0, file1, da_avg_freq, axis, angle):
    """
    Computes emissivity from two pamtra simulations and the observed
    brightness temperature dataset

    Parameters
    ----------
    file0: pamtra simulation file with emissivity of 0
    file1: pamtra simulation file with emissivity of 1
    da_e: emissivity value or dataset
    ds_tb: dataset with tb and incidence angle
    da_avg_freq: frequencies that will be averaged for each
      channel to map the frequency coordinate in pamtra to channel coordinate
      of the tb dataset
    axis: variable name that will be swapped with the default 'grid_x' (e.g.
      time or another variable that is contained in the pamtra simulation
      dataset)
    angle: name of incidence angle variable in tb dataset ds_tb

    Returns
    -------
    da_e: emissivity data array
    da_dtb: tb sensitivity to changes in emissivity
    """

    # prepare pamtra data
    ds_pam0, ds_pam1 = prepare_pamtra(
        file0, file1, ds_tb, da_avg_freq, axis, angle
    )

    ds_tb["tb_sim"] = tb_equation(e=da_e, tb_e0=ds_pam0.tb, tb_e1=ds_pam1.tb)

    return ds_tb


def tb_equation(e, tb_e0, tb_e1):
    """
    Tb calculation from pre-defined emissivity and two radiative transfer
    simulations.

    Parameters
    ----------
    e: emissivity
    tb_e0: brightness temperature under emissivity of 0
    tb_e1: brightness temperature under emissivity of 1

    Returns
    -------
    tb: observed brightness temperature
    """

    tb = e * (tb_e1 - tb_e0) + tb_e0

    return tb


if __name__ == "__main__":
    main()
