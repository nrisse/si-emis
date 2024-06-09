"""
Comparison between airborne and satellite emissivity estimates.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lizard.readers.band_pass import read_band_pass, read_band_pass_combination

from si_emis.data_preparation.airsat import airborne_filter
from si_emis.readers.airsat import read_airsat
from si_emis.readers.emissivity import read_emissivity_aircraft
from si_emis.style import *
from lizard.writers.figure_to_file import write_figure


FLIGHT_IDS = [
    "ACLOUD_P5_RF23",
    "ACLOUD_P5_RF25",
    "AFLUX_P5_RF08",
    "AFLUX_P5_RF14",
    "AFLUX_P5_RF15",
]

# this dict contains all combinations to compare mirac with satellites
COMPARE = {
    "89_all": [
        dict(ac=1, ins="AMSR2", sat="GCOM-W", sc=13, a0=0, a1=90),
        dict(ac=1, ins="AMSR2", sat="GCOM-W", sc=14, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F16", sc=17, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F17", sc=17, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F18", sc=17, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F16", sc=18, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F17", sc=18, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F18", sc=18, a0=0, a1=90),
        dict(ac=1, ins="ATMS", sat="SNPP", sc=16, a0=0, a1=30),
        dict(ac=1, ins="ATMS", sat="NOAA-20", sc=16, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-A", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-B", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-C", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="NOAA-19", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="NOAA-18", sc=1, a0=0, a1=30),
    ],
    "183_all": [
        dict(ac=7, ins="SSMIS", sat="DMSP-F16", sc=9, a0=0, a1=90),
        dict(ac=7, ins="SSMIS", sat="DMSP-F17", sc=9, a0=0, a1=90),
        dict(ac=7, ins="SSMIS", sat="DMSP-F18", sc=9, a0=0, a1=90),
        dict(ac=7, ins="ATMS", sat="SNPP", sc=18, a0=0, a1=30),
        dict(ac=7, ins="ATMS", sat="NOAA-20", sc=18, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-A", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-B", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-C", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-19", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-18", sc=5, a0=0, a1=30),
    ],
    "183_160_all": [
        dict(ac=7, ins="SSMIS", sat="DMSP-F16", sc=8, a0=0, a1=90),
        dict(ac=7, ins="SSMIS", sat="DMSP-F17", sc=8, a0=0, a1=90),
        dict(ac=7, ins="SSMIS", sat="DMSP-F18", sc=8, a0=0, a1=90),
        dict(ac=7, ins="ATMS", sat="SNPP", sc=17, a0=0, a1=30),
        dict(ac=7, ins="ATMS", sat="NOAA-20", sc=17, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-A", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-B", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-C", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-19", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-18", sc=2, a0=0, a1=30),
    ],
    "89_near_nadir": [
        dict(ac=1, ins="ATMS", sat="SNPP", sc=16, a0=0, a1=30),
        dict(ac=1, ins="ATMS", sat="NOAA-20", sc=16, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-A", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-B", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-C", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="NOAA-19", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="NOAA-18", sc=1, a0=0, a1=30),
    ],
    "183_near_nadir": [
        dict(ac=7, ins="ATMS", sat="SNPP", sc=18, a0=0, a1=30),
        dict(ac=7, ins="ATMS", sat="NOAA-20", sc=18, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-A", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-B", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-C", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-19", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-18", sc=5, a0=0, a1=30),
    ],
    "183_160_near_nadir": [
        dict(ac=7, ins="ATMS", sat="SNPP", sc=17, a0=0, a1=30),
        dict(ac=7, ins="ATMS", sat="NOAA-20", sc=17, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-A", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-B", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-C", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-19", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-18", sc=2, a0=0, a1=30),
    ],
    "89_mhs": [
        dict(ac=1, ins="MHS", sat="Metop-A", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-B", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="Metop-C", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="NOAA-19", sc=1, a0=0, a1=30),
        dict(ac=1, ins="MHS", sat="NOAA-18", sc=1, a0=0, a1=30),
    ],
    "183_mhs": [
        dict(ac=7, ins="MHS", sat="Metop-A", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-B", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-C", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-19", sc=5, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-18", sc=5, a0=0, a1=30),
    ],
    "183_160_mhs": [
        dict(ac=7, ins="MHS", sat="Metop-A", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-B", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="Metop-C", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-19", sc=2, a0=0, a1=30),
        dict(ac=7, ins="MHS", sat="NOAA-18", sc=2, a0=0, a1=30),
    ],
    "183_160_atms": [
        dict(ac=7, ins="ATMS", sat="SNPP", sc=17, a0=0, a1=30),
        dict(ac=7, ins="ATMS", sat="NOAA-20", sc=17, a0=0, a1=30),
    ],
    "89_atms": [
        dict(ac=1, ins="ATMS", sat="SNPP", sc=16, a0=0, a1=30),
        dict(ac=1, ins="ATMS", sat="NOAA-20", sc=16, a0=0, a1=30),
    ],
    "89_hpol": [
        dict(ac=1, ins="AMSR2", sat="GCOM-W", sc=14, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F16", sc=18, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F17", sc=18, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F18", sc=18, a0=0, a1=90),
    ],
    "89_amsr2v": [
        dict(ac=1, ins="AMSR2", sat="GCOM-W", sc=13, a0=0, a1=90),
    ],
    "89_amsr2h": [
        dict(ac=1, ins="AMSR2", sat="GCOM-W", sc=14, a0=0, a1=90),
    ],
    "89_ssmisv": [
        dict(ac=1, ins="SSMIS", sat="DMSP-F16", sc=17, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F17", sc=17, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F18", sc=17, a0=0, a1=90),
    ],
    "89_ssmish": [
        dict(ac=1, ins="SSMIS", sat="DMSP-F16", sc=18, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F17", sc=18, a0=0, a1=90),
        dict(ac=1, ins="SSMIS", sat="DMSP-F18", sc=18, a0=0, a1=90),
    ],
    "243_160_all": [
        dict(ac=8, ins="SSMIS", sat="DMSP-F16", sc=8, a0=0, a1=90),
        dict(ac=8, ins="SSMIS", sat="DMSP-F17", sc=8, a0=0, a1=90),
        dict(ac=8, ins="SSMIS", sat="DMSP-F18", sc=8, a0=0, a1=90),
        dict(ac=8, ins="ATMS", sat="SNPP", sc=17, a0=0, a1=30),
        dict(ac=8, ins="ATMS", sat="NOAA-20", sc=17, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-A", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-B", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-C", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="NOAA-19", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="NOAA-18", sc=2, a0=0, a1=30),
    ],
    "243_160_near_nadir": [
        dict(ac=8, ins="ATMS", sat="SNPP", sc=17, a0=0, a1=30),
        dict(ac=8, ins="ATMS", sat="NOAA-20", sc=17, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-A", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-B", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-C", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="NOAA-19", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="NOAA-18", sc=2, a0=0, a1=30),
    ],
    "243_160_mhs": [
        dict(ac=8, ins="MHS", sat="Metop-A", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-B", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="Metop-C", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="NOAA-19", sc=2, a0=0, a1=30),
        dict(ac=8, ins="MHS", sat="NOAA-18", sc=2, a0=0, a1=30),
    ],
    "243_160_hpol": [
        dict(ac=8, ins="SSMIS", sat="DMSP-F16", sc=8, a0=0, a1=90),
        dict(ac=8, ins="SSMIS", sat="DMSP-F17", sc=8, a0=0, a1=90),
        dict(ac=8, ins="SSMIS", sat="DMSP-F18", sc=8, a0=0, a1=90),
    ],
}

# channels for spectral variation
CHANNEL_SETUPS = {
    "MHS (QV)": dict(
        scs=[1, 2, 5],
        ins="MHS",
        sats=["Metop-A", "Metop-B", "Metop-C", "NOAA-19", "NOAA-18"],
        a0=0,
        a1=30,
    ),
    "ATMS (QV, QH)": dict(
        scs=[16, 17, 18], ins="ATMS", sats=["SNPP", "NOAA-20"], a0=0, a1=30
    ),
    "AMSR2 (V)": dict(scs=[13], ins="AMSR2", sats=["GCOM-W"], a0=0, a1=90),
    "AMSR2 (H)": dict(scs=[14], ins="AMSR2", sats=["GCOM-W"], a0=0, a1=90),
    "SSMIS (V)": dict(
        scs=[17],
        ins="SSMIS",
        sats=["DMSP-F16", "DMSP-F17", "DMSP-F18"],
        a0=0,
        a1=90,
    ),
    "SSMIS (H)": dict(
        scs=[18, 8, 9],
        ins="SSMIS",
        sats=["DMSP-F16", "DMSP-F17", "DMSP-F18"],
        a0=0,
        a1=90,
    ),
}

# channels for spectral variation
CHANNEL_SETUP_MIRAC = {
    "MiRAC": dict(acs=[1, 7, 8, 9], ins="MiRAC"),
}

AXIS_LIMITS = {"AFLUX_P5_RF08": {"ts": (245, 270)}}


def print_compare():
    """
    Prints the channels that will be compared
    """

    for name, setups in COMPARE.items():
        print(f"\nSetup {name}:")
        for i, setup in enumerate(setups):
            sat_label = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .label_pol.item()
            )
            ac_label = (
                read_band_pass_combination("MiRAC-A", "MiRAC-P")
                .sel(channel=setup["ac"])
                .label_pol.item()
            )

            print(
                f'{i}: {setup["ins"]}/{setup["sat"]} {sat_label} '
                f"vs {ac_label}"
            )


def print_channel_setups():
    """
    Print channel setups that are used for spectral emissivity variation
    """

    for name, setup in CHANNEL_SETUPS.items():
        for satellite in setup["sats"]:
            ds_bp = read_band_pass(setup["ins"], satellite)
            label_pol = ds_bp.label_pol.sel(channel=setup["scs"]).values
            print(f"{name} from {satellite}: {label_pol}")


def scatter_emissivity(
    da_es: xr.DataArray,
    da_ea_mean: xr.DataArray,
    da_sat_angle: xr.DataArray,
    da_satellite: xr.DataArray,
    da_instrument: xr.DataArray,
    da_channel: xr.DataArray,
    flight_id: str,
    surf_refl: str,
    comparison_89: Union[str, list] = None,
    comparison_183: Union[str, list] = None,
) -> None:
    """
    Scatter between emissivity from aircraft and emissivity from satellite.
    Corresponding channels from satellite and aircraft are pre-defined as well
    as restrictions on the satellite incidence angle.

    figure layout: (1, 2)
    left: 89 GHz
    right: 183 GHz
    x-axis: aircraft
    y-axis satellite
    colors: instrument+platform (e.g. MHS/Metop-B, ...)
    symbols: polarization (square: h-pol, triangle: v-pol, cross: qv-pol,
        plus: qh-pol)

    Parameters
    ----------
    da_es: xr.DataArray
        Emissivity from satellite as a function of satellite index and surface
        reflection type
    da_ea_mean: xr.DataArray
        Emissivity from aircraft averaged to satellite index as a function of
        satellite index, surface reflection type, and channel
    da_sat_angle: xr.DataArray
        Incidence angle corresponding to satellite index
    da_satellite: xr.DataArray
        Satellite platform corresponding to satellite index
    da_instrument: xr.DataArray
        Satellite instrument corresponding to satellite index
    da_channel: xr.DataArray
        Satellite channel corresponding to satellite index
    flight_id: str
        ac3airborne flight id
    surf_refl: str
        Surface reflection type
    comparison_89: Union[str, list], optional
        Defines the comparison setup for 89 GHz. Either use one pre-defined
        setup name or provide a list of dicts.
        Default is None.
    comparison_183: Union[str, list], optional
        Defines the comparison setup for 183 GHz. Either use one pre-defined
        setup name or provide a list of dicts.
        Default is None.
    errorbar: bool, optional
        Whether to plot error bar for observed emissivity. Default is True.
    """

    if comparison_89 is None:
        comparison_89 = "89_all"

    if comparison_183 is None:
        comparison_183 = "183_all"

    if isinstance(comparison_89, str):
        setups_89 = COMPARE[comparison_89]
    else:
        setups_89 = comparison_89

    if isinstance(comparison_183, str):
        setups_183 = COMPARE[comparison_183]
    else:
        setups_183 = comparison_183

    setups_lst = [setups_89, setups_183]

    # alpha level is proportional to the number of airborne footprints
    cnt_max = 200
    alpha_factor = 1 / cnt_max

    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharex="row", sharey="row")

    for i, ax in enumerate(axes):
        c = setups_lst[i]

        for setup in c:
            # select satellite indices matching the provided conditions
            ix = (
                (da_instrument == setup["ins"])
                & (da_satellite == setup["sat"])
                & (da_channel == setup["sc"])
                & (da_sat_angle >= setup["a0"])
                & (da_sat_angle < setup["a1"])
            )

            if np.sum(ix) == 0:
                print("skip")
                continue

            alpha = 1

            p = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .polarization.item()
            )

            label_pol = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .label_pol.item()
            )

            label = f'{setup["ins"]}/{setup["sat"]} {label_pol}'

            ax.scatter(
                da_ea_mean.sel(
                    surf_refl=surf_refl,
                    ac_channel=setup["ac"],
                    ix_sat_int=ix,
                ),
                da_es.sel(surf_refl=surf_refl, ix_sat_int=ix),
                color=platform_colors[setup["sat"]]["color"],
                label=label,
                marker=pol_markers[p]["marker"],
                linewidth=0,
                alpha=alpha,
            )

        ax.set_aspect("equal")
        ax.plot([0, 1], [0, 1], color="k", linewidth=0.75)

        leg = ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            fontsize=6,
            frameon=False,
        )
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    axes[0].set_xlim(0.6, 1)
    axes[0].set_ylim(0.6, 1)

    axes[0].set_xlabel("Emissivity MiRAC 89 (H) GHz")
    axes[1].set_xlabel(r"Emissivity MiRAC 183.31$\pm$7.5 (V) GHz")
    axes[0].set_ylabel("Emissivity Satellite")

    write_figure(
        fig,
        f"comparison/comparison_emissivity_{surf_refl}_" f"{flight_id}.png",
    )

    plt.close()


def scatter_emissivity_variability(
    da_es: xr.DataArray,
    da_ea_mean: xr.DataArray,
    da_ea_std: xr.DataArray,
    da_sat_angle: xr.DataArray,
    da_satellite: xr.DataArray,
    da_instrument: xr.DataArray,
    da_channel: xr.DataArray,
    flight_id: str,
    surf_refl: str,
    comparison_89: Union[str, list] = None,
    comparison_183: Union[str, list] = None,
) -> None:
    """
    Scatter between emissivity variability from aircraft and emissivity
    difference between satellite and aircraft. This provides an extension of
    the simple emissivity-emissivity scatter plot. The basic idea is that
    the difference between mean emissivity from aircraft and emissivity
    from satellite increases when the airborne-derived emissivity increases.

    Corresponding channels from satellite and aircraft are pre-defined as well
    as restrictions on the satellite incidence angle.

    figure layout: (1, 2)
    left: 89 GHz
    right: 183 GHz
    x-axis: aircraft
    y-axis satellite
    colors: instrument+platform (e.g. MHS/Metop-B, ...)
    symbols: polarization (square: h-pol, triangle: v-pol, cross: qv-pol,
        plus: qh-pol)

    Parameters
    ----------
    da_es: xr.DataArray
        Emissivity from satellite as a function of satellite index and surface
        reflection type
    da_ea_mean: xr.DataArray
        Emissivity from aircraft averaged to satellite index as a function of
        satellite index, surface reflection type, and channel
    da_ea_std: xr.DataArray
        Standard deviation of emissivity from aircraft within a satellite
        footprint, as a function of satellite index, surface reflection type,
        and channel
    da_sat_angle: xr.DataArray
        Incidence angle corresponding to satellite index
    da_satellite: xr.DataArray
        Satellite platform corresponding to satellite index
    da_instrument: xr.DataArray
        Satellite instrument corresponding to satellite index
    da_channel: xr.DataArray
        Satellite channel corresponding to satellite index
    flight_id: str
        ac3airborne flight id
    surf_refl: str
        Surface reflection type
    comparison_89: Union[str, list], optional
        Defines the comparison setup for 89 GHz. Either use one pre-defined
        setup name or provide a list of dicts.
        Default is None.
    comparison_183: Union[str, list], optional
        Defines the comparison setup for 183 GHz. Either use one pre-defined
        setup name or provide a list of dicts.
        Default is None.
    """

    if comparison_89 is None:
        comparison_89 = "89_all"

    if comparison_183 is None:
        comparison_183 = "183_all"

    if isinstance(comparison_89, str):
        setups_89 = COMPARE[comparison_89]
    else:
        setups_89 = comparison_89

    if isinstance(comparison_183, str):
        setups_183 = COMPARE[comparison_183]
    else:
        setups_183 = comparison_183

    setups_lst = [setups_89, setups_183]

    # alpha level is proportional to the number of airborne footprints
    cnt_max = 200
    alpha_factor = 1 / cnt_max

    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharex="row", sharey="row")

    for i, ax in enumerate(axes):
        c = setups_lst[i]

        for setup in c:
            # select satellite indices matching the provided conditions
            ix = (
                (da_instrument == setup["ins"])
                & (da_satellite == setup["sat"])
                & (da_channel == setup["sc"])
                & (da_sat_angle >= setup["a0"])
                & (da_sat_angle < setup["a1"])
            )

            print(
                f'{ix.sum("ix_sat_int").values.item()} matching footprints '
                f"with"
                f' {setup["ins"]}/{setup["sat"]} with several airborne '
                f"footprints"
            )

            if np.sum(ix) == 0:
                print("skip")
                continue

            alpha = 1

            p = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .polarization.item()
            )

            label_pol = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .label_pol.item()
            )

            label = f'{setup["ins"]}/{setup["sat"]} {label_pol}'

            ax.scatter(
                da_ea_std.sel(
                    surf_refl=surf_refl, ac_channel=setup["ac"], ix_sat_int=ix
                ),
                da_es.sel(surf_refl=surf_refl, ix_sat_int=ix)
                - da_ea_mean.sel(
                    surf_refl=surf_refl, ac_channel=setup["ac"], ix_sat_int=ix
                ),
                color=platform_colors[setup["sat"]]["color"],
                label=label,
                marker=pol_markers[p]["marker"],
                linewidth=0,
                alpha=alpha,
            )

        leg = ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            fontsize=6,
            frameon=False,
        )
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    # axes[0].set_xlim(0.6, 1)
    # axes[0].set_ylim(0.6, 1)

    axes[0].set_xlabel("Emissivity std MiRAC 89 (H) GHz")
    axes[1].set_xlabel(r"Emissivity std MiRAC 183.31$\pm$7.5 (V) GHz")
    axes[0].set_ylabel("Emissivity difference (satellite-MiRAC)")

    write_figure(
        fig,
        f"comparison/emissivity_variability_{surf_refl}_" f"{flight_id}.png",
    )

    plt.close()


def scatter_tb(
    da_es,
    da_ea_mean,
    da_sat_angle,
    da_satellite,
    da_instrument,
    da_channel,
    flight_id,
    surf_refl,
    comparison_89: Union[str, list] = None,
    comparison_183: Union[str, list] = None,
):
    """
    Scatterplot between tb from aircraft and tb from satellite.
    Corresponding channels from satellite and aircraft are pre-defined as well
    as restrictions on the satellite incidence angle.

    figure layout: (1, 2)
    left: 89 GHz
    right: 183 GHz
    x-axis: aircraft
    y-axis satellite
    colors: instrument+platform (e.g. MHS/Metop-B, ...)
    symbols: polarization (square: h-pol, triangle: v-pol, cross: qv-pol,
        plus: qh-pol)

    Parameters
    ----------
    da_es: tb from satellite as a function of satellite index
    da_ea_mean: tb from aircraft averaged to satellite index as a
      function of satellite index and channel
    da_sat_angle: incidence angle corresponding to satellite index
    da_satellite: satellite platform corresponding to satellite index
    da_instrument: satellite instrument corresponding to satellite index
    da_channel: satellite channel corresponding to satellite index
    flight_id: ac3airborne flight id
    surf_refl: surface reflection type
    comparison_89: defines the comparison setup for 89 GHz. Either use
        one pre-defined setup name or provide a list of dicts of the following
        structure:
        dict(ac=1, ins='MHS', sat='NOAA-19', sc=1, a0=0, a1=30)
    comparison_183: same, but for 183 GHz
    """

    if comparison_89 is None:
        comparison_89 = "89_all"

    if comparison_183 is None:
        comparison_183 = "183_all"

    if isinstance(comparison_89, str):
        setups_89 = COMPARE[comparison_89]
    else:
        setups_89 = comparison_89

    if isinstance(comparison_183, str):
        setups_183 = COMPARE[comparison_183]
    else:
        setups_183 = comparison_183

    setups_lst = [setups_89, setups_183]

    # alpha level is proportional to the number of airborne footprints
    cnt_max = 200
    alpha_factor = 1 / cnt_max

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))

    for i, ax in enumerate(axes):
        c = setups_lst[i]

        for setup in c:
            # select satellite indices matching the provided conditions
            ix = (
                (da_instrument == setup["ins"])
                & (da_satellite == setup["sat"])
                & (da_channel == setup["sc"])
                & (da_sat_angle >= setup["a0"])
                & (da_sat_angle < setup["a1"])
            )

            print(
                f'{ix.sum("ix_sat_int").values.item()} matching footprints '
                f"with"
                f' {setup["ins"]}/{setup["sat"]} with several airborne '
                f"footprints"
            )

            if np.sum(ix) == 0:
                print("skip")
                continue

            alpha = 1

            p = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .polarization.item()
            )

            label_pol = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .label_pol.item()
            )

            label = f'{setup["ins"]}/{setup["sat"]} {label_pol}'

            ax.scatter(
                da_ea_mean.sel(
                    ac_channel=setup["ac"], ix_sat_int=ix, surf_refl=surf_refl
                ),
                da_es.sel(ix_sat_int=ix, surf_refl=surf_refl),
                color=platform_colors[setup["sat"]]["color"],
                label=label,
                marker=pol_markers[p]["marker"],
                linewidth=0,
                alpha=alpha,
            )

        ax.set_aspect("equal")
        ax.plot([100, 300], [100, 300], color="k", linewidth=0.75)

        leg = ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            fontsize=6,
            frameon=False,
        )
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    axes[0].set_xlim(170, 280)
    axes[0].set_ylim(170, 280)

    axes[1].set_xlim(200, 250)
    axes[1].set_ylim(200, 250)

    axes[0].set_xlabel("TB MiRAC 89 (H) GHz")
    axes[1].set_xlabel(r"TB MiRAC 183.31$\pm$7.5 (V) GHz")
    axes[0].set_ylabel("TB Satellite")

    write_figure(fig, f"comparison/comparison_tb_{flight_id}.png")

    plt.close()


def scatter_ts(
    da_es,
    da_ea_mean,
    da_sat_angle,
    da_satellite,
    da_instrument,
    da_channel,
    flight_id,
    surf_refl,
    comparison_89: Union[str, list] = None,
    comparison_183: Union[str, list] = None,
):
    """
    Scatterplot between surface temperature from aircraft and surface
    temperature from ERA-5 or NEMO in the satellite footprint.
    Corresponding channels from satellite and aircraft are pre-defined as well
    as restrictions on the satellite incidence angle.

    Note that usually no difference between satellite channels of the same
    platform and granule occur because their lon and lat coordiantes are equal.

    figure layout: (1, 2)
    left: 89 GHz
    right: 183 GHz
    x-axis: aircraft
    y-axis satellite
    colors: instrument+platform (e.g. MHS/Metop-B, ...)
    symbols: polarization (square: h-pol, triangle: v-pol, cross: qv-pol,
        plus: qh-pol)

    Parameters
    ----------
    da_es: ts at satellite footprint as a function of satellite index
    da_ea_mean: ts from aircraft averaged to satellite index as a
      function of satellite index and channel
    da_sat_angle: incidence angle corresponding to satellite index
    da_satellite: satellite platform corresponding to satellite index
    da_instrument: satellite instrument corresponding to satellite index
    da_channel: satellite channel corresponding to satellite index
    flight_id: ac3airborne flight id
    surf_refl: surface reflection type
    comparison_89: defines the comparison setup for 89 GHz. Either use
        one pre-defined setup name or provide a list of dicts of the following
        structure:
        dict(ac=1, ins='MHS', sat='NOAA-19', sc=1, a0=0, a1=30)
    comparison_183: same, but for 183 GHz
    """

    if comparison_89 is None:
        comparison_89 = "89_all"

    if comparison_183 is None:
        comparison_183 = "183_all"

    if isinstance(comparison_89, str):
        setups_89 = COMPARE[comparison_89]
    else:
        setups_89 = comparison_89

    if isinstance(comparison_183, str):
        setups_183 = COMPARE[comparison_183]
    else:
        setups_183 = comparison_183

    setups_lst = [setups_89, setups_183]

    # alpha level is proportional to the number of airborne footprints
    cnt_max = 200
    alpha_factor = 1 / cnt_max

    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharex="all", sharey="all")

    for i, ax in enumerate(axes):
        c = setups_lst[i]

        for setup in c:
            # select satellite indices matching the provided conditions
            ix = (
                (da_instrument == setup["ins"])
                & (da_satellite == setup["sat"])
                & (da_channel == setup["sc"])
                & (da_sat_angle >= setup["a0"])
                & (da_sat_angle < setup["a1"])
            )

            print(
                f'{ix.sum("ix_sat_int").values.item()} matching footprints '
                f"with"
                f' {setup["ins"]}/{setup["sat"]} with several airborne '
                f"footprints"
            )

            if np.sum(ix) == 0:
                print("skip")
                continue

            alpha = 1

            p = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .polarization.item()
            )

            label_pol = (
                read_band_pass(instrument=setup["ins"], satellite=setup["sat"])
                .sel(channel=setup["sc"])
                .label_pol.item()
            )

            label = f'{setup["ins"]}/{setup["sat"]} {label_pol}'

            ax.scatter(
                da_ea_mean.sel(
                    ac_channel=setup["ac"], ix_sat_int=ix, surf_refl=surf_refl
                ),
                da_es.sel(ix_sat_int=ix, surf_refl=surf_refl),
                color=platform_colors[setup["sat"]]["color"],
                label=label,
                marker=pol_markers[p]["marker"],
                linewidth=0,
                alpha=alpha,
            )

        ax.set_aspect("equal")
        ax.plot([200, 300], [200, 300], color="k", linewidth=0.75)

        leg = ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            fontsize=6,
            frameon=False,
        )
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    limits = AXIS_LIMITS.get(flight_id)
    if limits:
        axes[0].set_xlim(limits["ts"])
        axes[0].set_ylim(limits["ts"])

        axes[1].set_xlim(limits["ts"])
        axes[1].set_ylim(limits["ts"])

    else:
        axes[0].set_xlim(230, 280)
        axes[0].set_ylim(230, 280)

        axes[1].set_xlim(230, 280)
        axes[1].set_ylim(230, 280)

    axes[0].set_xlabel("Surface temperature at MiRAC 89 (H) GHz")
    axes[1].set_xlabel(r"Surface temperature at MiRAC 183.31$\pm$7.5 (V) GHz")
    axes[0].set_ylabel("Surface temperature at satellite")

    write_figure(fig, f"comparison/comparison_ts_{flight_id}.png")

    plt.close()


def main(flight_id, comparison_89, comparison_183, surf_refl):
    """
    Runs a set of plotting rountines.

    Parameters
    ----------
    flight_id : str
        Flight identifier.
    comparison_89 : str
        Emissivity comparison on channel 89.
    comparison_183 : str
        Emissivity comparison on channel 183.
    surf_refl : str
        Surface reflection (L or S).
    """

    # read satellite footprints with aircraft resampled to satellite
    ds_es = read_airsat(flight_id=flight_id)

    # read aircraft emissivity at high resolution
    ds_ea = read_emissivity_aircraft(flight_id=flight_id)
    ds_ea = airborne_filter(ds_ea, drop_times=True)

    # emissivity
    scatter_emissivity(
        da_es=ds_es.e,
        da_ea_mean=ds_es["ea_e_mean"],
        da_sat_angle=ds_es.incidence_angle,
        da_satellite=ds_es.satellite,
        da_instrument=ds_es.instrument,
        da_channel=ds_es.channel,
        flight_id=flight_id,
        surf_refl=surf_refl,
        comparison_89=comparison_89,
        comparison_183=comparison_183,
    )

    # emissivity variability
    scatter_emissivity_variability(
        da_es=ds_es.e,
        da_ea_mean=ds_es["ea_e_mean"],
        da_ea_std=ds_es["ea_e_std"],
        da_sat_angle=ds_es.incidence_angle,
        da_satellite=ds_es.satellite,
        da_instrument=ds_es.instrument,
        da_channel=ds_es.channel,
        flight_id=flight_id,
        surf_refl=surf_refl,
        comparison_89=comparison_89,
        comparison_183=comparison_183,
    )

    # brightness temperature
    scatter_tb(
        da_es=ds_es.tb_sat,
        da_ea_mean=ds_es["ea_tb_mean"],
        da_sat_angle=ds_es.incidence_angle,
        da_satellite=ds_es.satellite,
        da_instrument=ds_es.instrument,
        da_channel=ds_es.channel,
        surf_refl=surf_refl,
        flight_id=flight_id,
        comparison_89=comparison_89,
        comparison_183=comparison_183,
    )

    # surface temperature
    scatter_ts(
        da_es=ds_es.ts,
        da_ea_mean=ds_es["ea_ts_mean"],
        da_sat_angle=ds_es.incidence_angle,
        da_satellite=ds_es.satellite,
        da_instrument=ds_es.instrument,
        da_channel=ds_es.channel,
        flight_id=flight_id,
        surf_refl=surf_refl,
        comparison_89=comparison_89,
        comparison_183=comparison_183,
    )


if __name__ == "__main__":

    for flight_id in FLIGHT_IDS:
        main(
            flight_id=flight_id,
            comparison_89="89_all",
            comparison_183="183_160_all",
            surf_refl="L",
        )
