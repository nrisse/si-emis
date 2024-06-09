"""
Creates boxplot of emissivity for all instruments and channels.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lizard.readers.band_pass import read_band_pass
from lizard.readers.emissivity import read_emissivity_aircraft
from lizard.rt.polarization import vh2qv
from lizard.writers.figure_to_file import write_figure

from si_emis.analysis.emissivity_comparison import (
    CHANNEL_SETUP_MIRAC,
    CHANNEL_SETUPS,
)
from si_emis.data_preparation.airsat import airborne_filter
from si_emis.readers.airsat import read_airsat
from si_emis.style import sensor_colors


def calculate_qv_emissivity(
    ds_es_merge, instrument, surf_refl, print_ratio=True
):
    """
    Calculated emissivity with QV mixed polarization at 89 GHz for AMSR2 and
    SSMIS. This is then comparable to MHS and ATMS. It could be extended
    for QH as well.

    Parameters
    ----------
    ds_es_merge : xr.Dataset
        merged emissivity dataset from satellite
    instrument : str
        instrument name
    surf_refl : str
        surface reflection type
    print_ratio : bool
        whether to print the ratio of V-pol in QV
    """

    # channel definitions for vertical and horizontal for SSMIS and AMSR2
    combinations = {
        "SSMIS": {
            "v": 17,
            "h": 18,
        },
        "AMSR2": {
            "v": 13,
            "h": 14,
        },
    }

    ev = query_satellite(
        ds_es_merge, instrument, combinations[instrument]["v"], 0, 90
    ).e.sel(surf_refl=surf_refl)
    eh = query_satellite(
        ds_es_merge, instrument, combinations[instrument]["h"], 0, 90
    ).e.sel(surf_refl=surf_refl)
    incidence_angle = query_satellite(
        ds_es_merge, instrument, combinations[instrument]["v"], 0, 90
    ).incidence_angle
    alt = query_satellite(
        ds_es_merge, instrument, combinations[instrument]["v"], 0, 90
    ).alt_sat

    da_e_qv = vh2qv(
        v=ev,
        h=eh.values,
        incidence_angle=incidence_angle,
        alt=alt,
    )

    if print_ratio:
        ratio = (
            vh2qv(
                v=0,
                h=1,
                incidence_angle=incidence_angle,
                alt=alt,
            )
            .mean()
            .item()
        )
        print(f"{instrument} ratio of H-pol in QV: {ratio}")

    return da_e_qv


def spectral_distribution_campaign(ax, surf_refl, flight_ids):
    """
    Spectral emissivity variation as violin plot.

    The plot requires a lot of manual channel selection and arrangement.

    Parameters
    ----------
    ax: axis to plot on
    surf_refl: surface reflection type
    flight_ids: flight ids to be plotted together
    """

    # read data from all flights
    ds_ea_merge, ds_es_merge = read_all_data(flight_ids)

    # airborne emissivity averaged to satellite footprint
    mirac_89_mhs = query_satellite(ds_es_merge, "MHS", 1, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=1)
    mirac_89_atms = query_satellite(ds_es_merge, "ATMS", 16, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=1)
    mirac_89_ssmis = query_satellite(ds_es_merge, "SSMIS", 17, 0, 90)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=1)
    mirac_89_amsr2 = query_satellite(ds_es_merge, "AMSR2", 13, 0, 90)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=1)

    mirac_183_mhs = query_satellite(ds_es_merge, "MHS", 2, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=7)
    mirac_183_atms = query_satellite(ds_es_merge, "ATMS", 17, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=7)
    mirac_183_ssmis = query_satellite(ds_es_merge, "SSMIS", 9, 0, 90)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=7)

    mirac_243_mhs = query_satellite(ds_es_merge, "MHS", 2, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=8)
    mirac_243_atms = query_satellite(ds_es_merge, "ATMS", 17, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=8)
    mirac_243_ssmis = query_satellite(ds_es_merge, "SSMIS", 9, 0, 90)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=8)

    mirac_340_mhs = query_satellite(ds_es_merge, "MHS", 2, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=9)
    mirac_340_atms = query_satellite(ds_es_merge, "ATMS", 17, 0, 30)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=9)
    mirac_340_ssmis = query_satellite(ds_es_merge, "SSMIS", 9, 0, 90)[
        "ea_e_mean"
    ].sel(surf_refl=surf_refl, ac_channel=9)

    data = [
        # 89 GHz
        query_satellite(ds_es_merge, "MHS", 1, 0, 30).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "ATMS", 16, 0, 30).e.sel(
            surf_refl=surf_refl
        ),
        calculate_qv_emissivity(ds_es_merge, "SSMIS", surf_refl),
        calculate_qv_emissivity(ds_es_merge, "AMSR2", surf_refl),
        query_satellite(ds_es_merge, "SSMIS", 17, 0, 90).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "AMSR2", 13, 0, 90).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "SSMIS", 18, 0, 90).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "AMSR2", 14, 0, 90).e.sel(
            surf_refl=surf_refl
        ),
        mirac_89_mhs,
        mirac_89_atms,
        mirac_89_ssmis,
        mirac_89_amsr2,
        # 160 GHz
        query_satellite(ds_es_merge, "MHS", 2, 0, 30).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "ATMS", 17, 0, 30).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "SSMIS", 8, 0, 90).e.sel(
            surf_refl=surf_refl
        ),
        # 183
        query_satellite(ds_es_merge, "MHS", 5, 0, 30).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "ATMS", 18, 0, 30).e.sel(
            surf_refl=surf_refl
        ),
        query_satellite(ds_es_merge, "SSMIS", 9, 0, 90).e.sel(
            surf_refl=surf_refl
        ),
        mirac_183_mhs,
        mirac_183_atms,
        mirac_183_ssmis,
        mirac_243_mhs,
        mirac_243_atms,
        mirac_243_ssmis,
        mirac_340_mhs,
        mirac_340_atms,
        mirac_340_ssmis,
    ]

    # print number of samples for each of the data sets
    for i in range(len(data)):
        print(len(data[i]))

    # remove nan's
    data = [x.where(~x.isnull(), drop=True) for x in data]

    # define instrument/emissivity, polarization, instrument/footprint
    pos_dict = {
        1: ("MHS", "QV", "MHS"),
        2: ("ATMS", "QV", "ATMS"),
        3: ("SSMIS", "QV", "SSMIS"),
        4: ("AMSR2", "QV", "AMSR2"),
        5: ("SSMIS", "V", "SSMIS"),
        6: ("AMSR2", "V", "AMSR2"),
        7: ("SSMIS", "H", "SSMIS"),
        8: ("AMSR2", "H", "AMSR2"),
        8.7: ("MiRAC", "H", "MHS"),
        8.9: ("MiRAC", "H", "ATMS"),
        9.1: ("MiRAC", "H", "SSMIS"),
        9.3: ("MiRAC", "H", "AMSR2"),
        11: ("MHS", "QV", "MHS"),
        12: ("ATMS", "QH", "ATMS"),
        13: ("SSMIS", "H", "SSMIS"),
        15: ("MHS", "QV", "MHS"),
        16: ("ATMS", "QH", "ATMS"),
        17: ("SSMIS", "H", "SSMIS"),
        17.7: ("MiRAC", "", "MHS"),
        17.9: ("MiRAC", "", "ATMS"),
        18.1: ("MiRAC", "", "SSMIS"),
        19.7: ("MiRAC", "", "MHS"),
        19.9: ("MiRAC", "", "ATMS"),
        20.1: ("MiRAC", "", "SSMIS"),
        21.7: ("MiRAC", "", "MHS"),
        21.9: ("MiRAC", "", "ATMS"),
        22.1: ("MiRAC", "", "SSMIS"),
    }
    positions = list(pos_dict.keys())
    instruments = [pos_dict[i][0] for i in positions]
    polarizations = [pos_dict[i][1] for i in positions]
    instruments_footprint = [pos_dict[i][2] for i in positions]
    label_list = [
        f"{i} {p}".strip() for i, p in zip(instruments, polarizations)
    ]

    # width is 0.8, only where mirac is it is 0.2
    widths = [0.8] * len(data)
    for i in range(len(data)):
        if instruments[i] == "MiRAC":
            widths[i] = 0.2

    # get list of gaps between bands
    position_gaps = []
    for i in range(len(positions) - 1):
        if (positions[i + 1] - positions[i]) > 1.1:
            position_gaps.append((positions[i] + positions[i + 1]) / 2)

    max_pos = max(positions)

    # this removes those channels without data, while keeping the positions
    # equal. e.g. no 183 GHz emissivity from satellite due to low sensitivity
    keep_index = []
    for i in range(len(data)):
        print(len(data[i]))
        if len(data[i].values) > 10:  # at least 10 observations needed
            keep_index.append(i)
    data = [data[i] for i in keep_index]
    label_list = [label_list[i] for i in keep_index]
    positions = [positions[i] for i in keep_index]
    instruments = [instruments[i] for i in keep_index]
    instruments_footprint = [instruments_footprint[i] for i in keep_index]
    widths = [widths[i] for i in keep_index]

    # remove labels from list where instrument is mirac and instrument footprint is not mhs
    label_list_orig = label_list.copy()
    label_list = []
    for i in range(len(label_list_orig)):
        if instruments[i] != "MiRAC":
            label_list.append(label_list_orig[i])
        elif instruments[i] == "MiRAC" and instruments_footprint[i] == "MHS":
            label_list.append(label_list_orig[i])

    # boxplot
    bpl = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="d",
            markersize=0.5,
            markerfacecolor="#555555",
            markeredgecolor="#555555",
        ),
        medianprops=dict(color="#555555"),
        boxprops=dict(color="#555555"),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
    )

    # box color depends on the footprint instrument (MiRAC depends on footprint)
    for i, pc in enumerate(bpl["boxes"]):
        pc.set_facecolor(sensor_colors[instruments_footprint[i]])

    # bar plot with number of count for each box on secondary axis
    ax2 = ax.twinx()
    ax2.bar(
        positions,
        [len(d) for d in data],
        width=widths,
        color=[sensor_colors[i] for i in instruments_footprint],
        edgecolor="None",
    )
    ax2.set_ylim(0, 1500)
    ax2.set_yticks(np.arange(0, 301, 150))
    ax2.set_yticks(np.arange(0, 301, 50), minor=True)
    ax2.set_ylabel("Count", loc="bottom")

    # set ticklabels as sensor and polarization
    # this is adapted manually to have only one MiRAC label for the 3-4 boxes
    ticks, ix_unique_pos = np.unique(
        np.round(np.array(positions)).astype(int), return_index=True
    )  # removes the float positions
    ax.set_xticks(ticks)

    # remove labels
    ax.set_xticklabels(label_list, rotation=90, ha="center")
    for i, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color(sensor_colors[instruments[ix_unique_pos[i]]])

    ax.set_yticks(np.arange(0.5, 1.1, 0.1))

    # ticklabels closer to axis
    ax.tick_params(axis="x", which="major", pad=0)

    # annotate frequency range of sensors
    dy = 1
    ax.annotate(
        "88$-$92",
        xy=(5, dy),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
    )
    ax.annotate(
        "150$-$165",
        xy=(12, dy),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
    )
    ax.annotate(
        "176$-$190",
        xy=(16.5, dy),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
    )
    ax.annotate(
        "243",
        xy=(20, dy),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
    )
    ax.annotate(
        "340",
        xy=(22, dy),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
    )
    ax.annotate(
        "[GHz]",
        xy=(1.02, dy),
        xycoords="axes fraction",
        ha="left",
        va="top",
    )

    ax.set_ylabel("Emissivity")

    for x in position_gaps:
        ax.axvline(x, color="k", linewidth=0.75)

    ax.set_ylim(0.5, 1)
    ax.set_xlim(0, max_pos + 1)

    # make grid with horizontal lines
    ax.grid(axis="y")

    # print median emissivity for every dataset
    for i in range(len(data)):
        print(f"{label_list_orig[i]} {np.median(data[i])}")

    # print standard deviation for every dataset
    for i in range(len(data)):
        print(f"{label_list_orig[i]} {np.std(data[i].values)}")


def query_satellite(ds_es, instrument, channel, angle0, angle1):
    """
    Queries satellite dataset for specific instrument. Note that all satellites
    are included here.

    The function also accepts list of instruments, channels, and angles. Then,
    it returns True if the footprint matches at least one of the combinations.

    Parameters
    ----------
    ds_es: emissivity dataset from satellite
    instrument: instrument name
    channel: channel name
    angle0: lowest angle
    angle1: highest angle

    Returns
    -------
    ds_es_qry: queried dataset
    """

    if isinstance(instrument, list):
        ix = satellite_index_multiple(
            ds_es, instrument, channel, angle0, angle1
        )
        ds_es_qry = ds_es.sel(ix_sat=ix)

    else:
        ix = satellite_index(ds_es, instrument, channel, angle0, angle1)
        ds_es_qry = ds_es.sel(ix_sat=ix)

    return ds_es_qry


def read_all_data(flight_ids):
    """
    This function contains all the magic to have matching airborne and
    satellite observations. Just provide a list of flight ids.

    The following steps are included here:
    - distance threshold, i.e., keep only pairs of aircraft/satellite within a
      certain distance
    - hour threshold, i.e., keep only pairs of aircraft/satellite within a
      certain time
    - keep only pairs where at least 10 satellite footprints align with an
      airborne footprint

    The output can be used also to get PAMTRA forward simulations for either
    dataset by taking the ix_sat_int or ix_sat variables. Note that ix_sat_int
    is only unique for each research flight.

    Parameters
    ----------
    surf_refl: surface reflection type
    flight_ids: flight ids to be plotted together

    Returns
    -------
    ds_ea_merge: airborne emissivity dataset for the selected flights
    ds_es_merge: satellite emissivity for the flight
    """

    lst_ds_ea = []
    lst_ds_es = []
    for flight_id in flight_ids:
        # read satellite emissivity
        ds_es = read_airsat(flight_id=flight_id)

        # read aircraft emissivity
        ds_ea = read_emissivity_aircraft(flight_id=flight_id)
        ds_ea = airborne_filter(ds_ea, drop_times=True)

        lst_ds_ea.append(ds_ea)

        # set string as index
        ds_es = ds_es.swap_dims({"ix_sat_int": "ix_sat"})

        lst_ds_es.append(ds_es)

    # concatenate to single dataset
    ds_ea_merge = xr.concat(lst_ds_ea, dim="time")
    ds_es_merge = xr.concat(lst_ds_es, dim="ix_sat")

    return ds_ea_merge, ds_es_merge


def satellite_index_multiple(ds_es, instrument, channel, angle0, angle1):
    """
    Extension of satellite_index() by allowing to query for multiple
    instruments and angles at once. This is useful if one wants all MiRAC
    observations averaged to the footprints of more than one satellite.

    All inputs should be given as a list and are evaluated element-wise.

    Parameters
    ----------
    ds_es: emissivity dataset from satellite
    instrument: list of instrument names
    channel: list of channel names
    angle0: list of lowest angles
    angle1: list of highest angles

    Returns
    -------
    ix: boolean mask of indices
    """

    ixs = []
    for i in range(len(instrument)):
        ix = satellite_index(
            ds_es, instrument[i], channel[i], angle0[i], angle1[i]
        )
        ixs.append(ix)

    ix = np.any(ixs, axis=0)

    return ix


def satellite_index(ds_es, instrument, channel, angle0, angle1):
    """
    Returns indices matching specific combination of instrument, channel, and
    angle.

    ds_es should contain arrays of instrument, channel, and incidence_angle

    Parameters
    ----------
    ds_es: emissivity dataset from satellite
    instrument: instrument name
    channel: channel name
    angle0: lowest angle
    angle1: highest angle

    Returns
    -------
    ix: boolean mask of indices
    """

    ix = (
        (ds_es.instrument == instrument)
        & (ds_es.channel == channel)
        & (ds_es.incidence_angle >= angle0)
        & (ds_es.incidence_angle < angle1)
    )

    return ix


def spectral_distribution():
    """
    Merges spectral dist plot from ACLOUD and AFLUX campaign into a single
    figure.
    """

    plt.rcParams["axes.spines.right"] = True

    fig, axes = plt.subplots(
        2, 1, figsize=(6, 5), sharey="all", constrained_layout=True
    )

    axes[0].annotate(
        "(a) ACLOUD",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )
    axes[1].annotate(
        "(b) AFLUX",
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
    )

    spectral_distribution_campaign(
        ax=axes[0],
        flight_ids=["ACLOUD_P5_RF23", "ACLOUD_P5_RF25"],
        surf_refl="L",
    )
    spectral_distribution_campaign(
        ax=axes[1],
        flight_ids=["AFLUX_P5_RF08", "AFLUX_P5_RF14", "AFLUX_P5_RF15"],
        surf_refl="L",
    )

    # add legend for the sensor colors
    sensors = ["MHS", "ATMS", "SSMIS", "AMSR2"]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=sensor_colors[i]) for i in sensors
    ]
    axes[0].legend(
        handles,
        sensors,
        loc="lower center",
        frameon=True,
        bbox_to_anchor=(0.7125, 0),
        ncol=1,
    )

    write_figure(fig, f"spectral_violin_L.png")

    plt.close()


if __name__ == "__main__":
    spectral_distribution()
