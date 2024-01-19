"""
Identify time shift between MiRAC-P and KT-19 from lagged correlations for
clear-sky segments only.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lizard.readers.bbr import read_bbr
from lizard.writers.figure_to_file import write_figure

from si_emis.readers.airborne_tb import read_from_intake
from si_emis.retrieval.makesetups import flight_ids, times
from si_emis.retrieval.pamsim import select_times
from si_emis.style import flight_colors


def main():
    """
    Compute correlations
    """

    shifts = np.arange(-10, 11)

    correlations = np.array([]).reshape(0, 21)
    optimal_shift = np.array([])
    corr_maxs = np.array([])

    for flight_id in flight_ids:
        ds_tb = read_from_intake(flight_id)
        da_ts = read_bbr(flight_id).KT19

        # reduce to clear-sky times
        ds_tb = select_times(ds_tb, times=times[flight_id])

        # mirac-p and kt-19
        opt_shift, corrs, shifts, corr_max = lagged_correlation(
            da1=ds_tb.tb.sel(channel=7), da2=da_ts, shifts=shifts
        )
        correlations = np.vstack([correlations, corrs])
        optimal_shift = np.append(optimal_shift, opt_shift)
        corr_maxs = np.append(corr_maxs, corr_max)

    # store result in dataset
    ds_shift = xr.Dataset()
    ds_shift.coords["t_shift"] = shifts
    ds_shift.coords["flight_id"] = flight_ids
    ds_shift["correlation"] = (("flight_id", "shift"), correlations)
    ds_shift["opt_shift"] = ("flight_id", optimal_shift)
    ds_shift["corr_max"] = ("flight_id", corr_maxs)

    plot_lc(ds_shift)


def lagged_correlation(da1, da2, shifts):
    """
    Computes lagged correlation between two data arrays.
    """

    corrs = np.array([])
    for shift in shifts:
        da1_shifted = da1.shift({"time": shift}, fill_value=np.nan).copy()
        corr = xr.corr(da1_shifted, da2).item()
        corrs = np.append(corrs, corr)

    ix_max = np.argmax(np.abs(corrs))
    opt_shift = shifts[ix_max]
    corr_max = np.max(np.abs(corrs))

    return opt_shift, corrs, shifts, corr_max


def plot_lc(ds_shift):
    """
    Plot lagged correlations
    """

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.axvline(x=0, color="gray", zorder=0)

    for flight_id in ds_shift.flight_id.values:
        ax.plot(
            ds_shift.t_shift,
            ds_shift.correlation.sel(flight_id=flight_id),
            label=flight_id.split("_")[-1],
            color=flight_colors[flight_id],
        )
        ax.scatter(
            ds_shift.opt_shift.sel(flight_id=flight_id),
            ds_shift.corr_max.sel(flight_id=flight_id),
            color=flight_colors[flight_id],
            s=20,
            lw=0,
        )

    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=4,
        bbox_to_anchor=(0.5, 1.3),
        loc="upper center",
    )

    ax.set_xlim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xticks(np.arange(-10, 11, 1), minor=True)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Time lags [s]")
    ax.set_ylabel(r"Corr($T_{b,243 \mathrm{GHz}}, T_s$)")

    ax.grid(which="both")

    write_figure(fig, "time_shift.png")

    plt.close()


if __name__ == "__main__":
    main()
