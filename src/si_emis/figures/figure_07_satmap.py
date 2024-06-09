"""
Airborne and satellite emissivity on map
"""

import string
    
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from lizard.readers.worldview import read_worldview
from lizard.satellite import IFOV
from matplotlib import cm

from si_emis.data_preparation.airsat import airborne_filter
from si_emis.readers.airsat import read_airsat
from si_emis.readers.emissivity import read_emissivity_aircraft
from si_emis.style import *
from si_emis.analysis.emissivity_comparison import COMPARE
from lizard.writers.figure_to_file import write_figure

MAP_EXTENTS = {
    "ACLOUD_P5_RF23": (-18614, 221067, -1226634, -893742),
    "ACLOUD_P5_RF25": (67670, 144137, -1139393, -1033189),
    "AFLUX_P5_RF08": (-158434, -34696, -1141479, -969620),
    "AFLUX_P5_RF14": (-83222, 7472, -1085922, -959958),
    "AFLUX_P5_RF15": (-75094, 95372, -1162515, -925756),
}

DLAT = {
    "ACLOUD_P5_RF23": 1,
    "ACLOUD_P5_RF25": 0.5,
    "AFLUX_P5_RF08": 0.5,
    "AFLUX_P5_RF14": 0.5,
    "AFLUX_P5_RF15": 0.5,
}

DLON = {
    "ACLOUD_P5_RF23": 5,
    "ACLOUD_P5_RF25": 2,
    "AFLUX_P5_RF08": 2,
    "AFLUX_P5_RF14": 2,
    "AFLUX_P5_RF15": 4,
}

SCALE_PARAMS = {
    "ACLOUD_P5_RF23": dict(width=80, step=20),
    "ACLOUD_P5_RF25": dict(width=30, step=5),
    "AFLUX_P5_RF08": dict(width=60, step=15),
    "AFLUX_P5_RF14": dict(width=40, step=10),
    "AFLUX_P5_RF15": dict(width=80, step=20),
}

CRS_PROJ = ccrs.NorthPolarStereo(central_longitude=10)

# colorscale for emissivity
CMAP_E = cmc.batlow
NORM_E = mcolors.BoundaryNorm(
    np.arange(0.65, 1 + 0.025, 0.025), ncolors=CMAP_E.N
)
TICKS_E = np.arange(0.6, 1.1, 0.1)

# colorscale for emissivity difference
CMAP_D = cmc.berlin
NORM_D = mcolors.BoundaryNorm(
    np.arange(-0.1, 0.1 + 0.025, 0.025), ncolors=CMAP_D.N
)
TICKS_D = np.arange(-0.1, 0.11, 0.1)

# colorscale for emissivity iqr in satellite footprint
CMAP_S = cmc.lipari
NORM_S = mcolors.BoundaryNorm(
    np.arange(0, 0.14 + 0.01, 0.02), ncolors=CMAP_S.N
)
TICKS_S = np.arange(0, 0.14 + 0.01, 0.07)

MAP_CONTENTS = {
    "MHS_ATMS_89": {
        "compare": [
            ["ACLOUD_P5_RF23", "89_near_nadir", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF08", "89_near_nadir", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF14", "89_near_nadir", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF15", "89_near_nadir", "1", "89 GHz", "89 GHz"],
        ],
        "sat_header": "MHS/ATMS",
    },
    "MHS_89": {
        "compare": [
            ["ACLOUD_P5_RF23", "89_mhs", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF08", "89_mhs", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF14", "89_mhs", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF15", "89_mhs", "1", "89 GHz", "89 GHz"],
        ],
        "sat_header": "MHS",
    },
    "AMSR2_89H": {
        "compare": [
            ["ACLOUD_P5_RF23", "89_amsr2h", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF08", "89_amsr2h", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF14", "89_amsr2h", "1", "89 GHz", "89 GHz"],
            ["AFLUX_P5_RF15", "89_amsr2h", "1", "89 GHz", "89 GHz"],
        ],
        "sat_header": "AMSR2",
    },
    "MHS_ATMS_243_160_or_183_160": {
        "compare": [
            [
                "ACLOUD_P5_RF23",
                "243_160_near_nadir",
                "8",
                "243 GHz",
                "157 GHz",
            ],
            [
                "ACLOUD_P5_RF25",
                "243_160_near_nadir",
                "8",
                "243 GHz",
                "157 GHz",
            ],
            [
                "AFLUX_P5_RF08",
                "183_160_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                "157 GHz",
            ],
            [
                "AFLUX_P5_RF14",
                "183_160_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                "157 GHz",
            ],
            [
                "AFLUX_P5_RF15",
                "183_160_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                "157 GHz",
            ],
        ],
        "sat_header": "MHS/ATMS",
    },
    "MHS_183_160": {
        "compare": [
            [
                "AFLUX_P5_RF08",
                "183_160_mhs",
                "7",
                r"183$\pm$7.5 GHz",
                "157 GHz",
            ],
            [
                "AFLUX_P5_RF14",
                "183_160_mhs",
                "7",
                r"183$\pm$7.5 GHz",
                "157 GHz",
            ],
            [
                "AFLUX_P5_RF15",
                "183_160_mhs",
                "7",
                r"183$\pm$7.5 GHz",
                "157 GHz",
            ],
        ],
        "sat_header": "MHS",
    },
    "paper_MHS_ATMS_183_160": {
        "compare": [
            [
                "AFLUX_P5_RF08",
                "183_160_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                "157/165 GHz",
            ],
            [
                "AFLUX_P5_RF14",
                "183_160_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                "157/165 GHz",
            ],
            [
                "AFLUX_P5_RF15",
                "183_160_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                "157/165 GHz",
            ],
        ],
        "sat_header": "Sat.",
    },
    "MHS_ATMS_183_183": {
        "compare": [
            [
                "AFLUX_P5_RF08",
                "183_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                r"190/183$\pm$7 GHz",
            ],
            [
                "AFLUX_P5_RF14",
                "183_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                r"190/183$\pm$7 GHz",
            ],
            [
                "AFLUX_P5_RF15",
                "183_near_nadir",
                "7",
                r"183$\pm$7.5 GHz",
                r"190/183$\pm$7 GHz",
            ],
        ],
        "sat_header": "Sat.",
    },
    "AMSR2_89H_RF23": {
        "compare": [
            ["ACLOUD_P5_RF23", "89_amsr2h", "1", "89H GHz", "89H GHz"],
        ],
        "sat_header": "AMSR2",
    },
    "AMSR2_89V_RF23": {
        "compare": [
            ["ACLOUD_P5_RF23", "89_amsr2v", "1", "89H GHz", "89V GHz"],
        ],
        "sat_header": "AMSR2",
    },
    "SSMIS_89H_RF23": {
        "compare": [
            ["ACLOUD_P5_RF23", "89_ssmish", "1", "89H GHz", "89H GHz"],
        ],
        "sat_header": "SSMIS",
    },
    "SSMIS_89V_RF23": {
        "compare": [
            ["ACLOUD_P5_RF23", "89_ssmisv", "1", "89H GHz", "89V GHz"],
        ],
        "sat_header": "SSMIS",
    },
}


def main():
    """
    Creates map
    """

    to_show = [
        "paper_MHS_ATMS_183_160",
        "MHS_ATMS_183_183",
        "MHS_ATMS_89",
        "AMSR2_89H_RF23",
        "AMSR2_89V_RF23",
        "SSMIS_89H_RF23",
        "SSMIS_89V_RF23"
    ]

    # numbers indicate the index in the list above
    for i, i_content in enumerate(to_show):
        print(i)
        map_subplot(map_content=MAP_CONTENTS[i_content])


def map_subplot(map_content):
    """
    This creates a map subplot with rows being for a specific flight/comparison
    method and columns being fixed in the sat_on_map() method.

    Parameters
    ----------
    map_content: dict with keys "compare" and "sat_header". Compare is a list of
        lists with the following structure:
        [flight_id, comparison, aircraft channel, description]
        Sat_header is a list of strings with the satellite name used for the
        header.
    """

    nrows = len(map_content["compare"])
    h_w_ratio = 0.3
    width = 7
    fig, axes = plt.subplots(
        nrows,
        5,
        figsize=(width, h_w_ratio * width * nrows),
        subplot_kw=dict(projection=CRS_PROJ),
    )

    if nrows == 1:
        axes = np.array([axes])

    # annotate letters and numbers for easier referencing
    for i, ax in enumerate(axes.flat):
        ax.annotate(
            f"({string.ascii_lowercase[i]})",
            xy=(0.03, 0.98),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )

    # this is needed to make sure that the scatter sizes are as specified when
    # plotting them. Otherwise, the scatter size is affected by zooming.
    # first: prepare maps and sizes, than draw the figure, then add the scatter
    # plots
    for i, (
        flight_id,
        comparison,
        ac_channel,
        ac_channel_text,
        sat_channel_text,
    ) in enumerate(map_content["compare"]):

        # prepare the map
        prepare_map(flight_id, axes[i, :])

        # annotate flight id on left side
        axes[i, -1].annotate(
            flight_id.replace("_P5_", " "),
            xy=(1.02, 0.5),
            xycoords="axes fraction",
            ha="left",
            va="center",
            rotation=-90,
        )

    # add colorbar below the lowest row
    fig.colorbar(
        cm.ScalarMappable(norm=NORM_E, cmap=CMAP_E),
        ax=axes[:, :3],
        label=r"e",
        ticks=TICKS_E,
        extend="both",
        orientation="horizontal",
        shrink=0.29,
        pad=0.02,
    )
    fig.colorbar(
        cm.ScalarMappable(norm=NORM_D, cmap=CMAP_D),
        ax=axes[:, 3],
        label=r"$\Delta$e",
        ticks=TICKS_D,
        extend="both",
        orientation="horizontal",
        pad=0.02,
    )
    fig.colorbar(
        cm.ScalarMappable(norm=NORM_S, cmap=CMAP_S),
        ax=axes[:, 4],
        label="Interquartile range",
        ticks=TICKS_S,
        extend="max",
        orientation="horizontal",
        pad=0.02,
    )

    # annotate information on top of the first row
    kwds = dict(
        xy=(0.5, 1),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
    )
    axes[0, 0].annotate(f"MiRAC\n{ac_channel_text}", **kwds)
    axes[0, 1].annotate(
        "MiRAC$"
        r"\rightarrow$"
        f"{map_content['sat_header']}\n{ac_channel_text}",
        **kwds,
    )
    axes[0, 2].annotate(
        f"{map_content['sat_header']}\n{sat_channel_text}", **kwds
    )
    axes[0, 3].annotate(f"{map_content['sat_header']}-MiRAC", **kwds)
    axes[0, 4].annotate(
        "MiRAC"
        r"$\rightarrow$"
        f"{map_content['sat_header']}\n{ac_channel_text}",
        **kwds,
    )

    # draw figure
    fig.canvas.draw()

    for i, (
        flight_id,
        comparison,
        ac_channel,
        ac_channel_text,
        sat_channel_text,
    ) in enumerate(map_content["compare"]):
        sat_on_map(
            fig=fig,
            axes_row=axes[i, :],
            flight_id=flight_id,
            comparison=comparison,
        )

    # add dummy legend with flight track with a small frame
    axes[-1, 2].plot([], [], color="darkgray", label="Polar 5")
    leg = axes[-1, 2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
    )
    leg.set_in_layout(False)

    write_figure(
        fig,
        f"comparison_emissivity_L"
        "_map_composite_"
        f"{map_content['compare'][0][1]}_{map_content['compare'][0][2]}.png",
    )

    plt.close()


def prepare_map(flight_id, axes_row):
    """
    Plots everything in map except for scatter points

    Parameters
    ----------
    flight_id: str
        Flight id
    axes_row: list of axes
        Axes to plot on
    """

    ww_data, ww_extent = read_worldview(flight_id)

    draw_labels = {
        0: ["left", "bottom"],
        1: ["bottom"],
        2: ["bottom"],
        3: ["bottom"],
        4: ["bottom"],
    }

    for i, ax in enumerate(axes_row):
        ax.set_extent(MAP_EXTENTS[flight_id], crs=CRS_PROJ)

        # add gridlines
        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=draw_labels[i],
            xlocs=mticker.FixedLocator(np.arange(-50, 50, DLON[flight_id])),
            ylocs=mticker.FixedLocator(np.arange(75, 83, DLAT[flight_id])),
            x_inline=False,
            y_inline=False,
            rotate_labels=False,
            linewidth=0.25,
            color="k",
            alpha=0.5,
            zorder=6,
        )

        ax.imshow(ww_data, extent=ww_extent, origin="upper")


def sat_on_map(fig, axes_row, flight_id, comparison):
    """
    View satellite footprints that match aircraft on map

    Shown on map:
    - flight track in black
    - points similar to scatter plot that represent location of each footprint
      as well as the satellite and instrument

    - difference in emissivity at 89 GHz (and 183 GHz) for a specific satellite
      sensor together with the flight track of the aircraft
    - questions: spatial patterns? depends on distance to flight track?

    Parameters
    ----------
    comparison: defines the comparison setup for 89 and 183 GHz. Either use one
       pre-defined setup name or provide a list with dicts of the following
       structure:
       dict(ac=7, ins='MHS', sat='NOAA-19', sc=5, a0=0, a1=30)
    """

    ds_ea, ds_es = read_data(flight_id)

    ixs = []
    for setup in COMPARE[comparison]:
        ixs.append(
            (ds_es.instrument == setup["ins"])
            & (ds_es.satellite == setup["sat"])
            & (ds_es.channel == setup["sc"])
            & (ds_es.incidence_angle >= setup["a0"])
            & (ds_es.incidence_angle < setup["a1"])
        )
    ix = xr.concat(ixs, dim="x").any("x").compute()

    # skip if index is zero
    if ix.sum() == 0:
        print("skip because no matching footprints")
        return

    # compute size of scatter points on map from footprint diameters
    # approximation: mean of semi major/minor axis and at nadir for sounders
    d_sat_km = [
        np.mean(IFOV[i][c])
        for i, c in zip(
            ds_es.instrument.sel(ix_sat_int=ix).values,
            ds_es.channel.sel(ix_sat_int=ix).values,
        )
    ]
    d_sat_m = np.array(d_sat_km) * 1e3
    d_ac_m = 100

    # rescale aircraft to make it visible
    factor_ac = 50
    d_ac_m *= factor_ac

    s_sat = diameter_to_scattersize(ax=axes_row[0], diameter=d_sat_m)
    s_ac = diameter_to_scattersize(ax=axes_row[0], diameter=d_ac_m)

    for i, ax in enumerate(axes_row):
        # flight track where emissivity data is available
        ax.scatter(
            ds_ea.lon.sel(
                channel=setup["ac"],
                time=~ds_ea.e.sel(surf_refl="L").isnull().all("channel"),
            ),
            ds_ea.lat.sel(
                channel=setup["ac"],
                time=~ds_ea.e.sel(surf_refl="L").isnull().all("channel"),
            ),
            color="darkgray",
            s=1,
            lw=0,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    if setup["ins"] == "AMSR2":
        zorder_sat = 6
    else:
        zorder_sat = 1

    # aircraft high resolution
    im = axes_row[0].scatter(
        ds_ea.lon.sel(channel=setup["ac"]),
        ds_ea.lat.sel(channel=setup["ac"]),
        c=ds_ea.e.sel(channel=setup["ac"], surf_refl="L"),
        cmap=CMAP_E,
        norm=NORM_E,
        s=s_ac,
        lw=0,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

    # aircraft averaged to satellite
    im = axes_row[1].scatter(
        ds_es.lon_sat.sel(ix_sat_int=ix),
        ds_es.lat_sat.sel(ix_sat_int=ix),
        c=ds_es["ea_e_mean"].sel(
            ix_sat_int=ix, surf_refl="L", ac_channel=setup["ac"]
        ),
        cmap=CMAP_E,
        norm=NORM_E,
        s=s_sat,
        transform=ccrs.PlateCarree(),
        zorder=zorder_sat,
    )

    # satellite
    im = axes_row[2].scatter(
        ds_es.lon_sat.sel(ix_sat_int=ix),
        ds_es.lat_sat.sel(ix_sat_int=ix),
        c=ds_es.e.sel(ix_sat_int=ix, surf_refl="L"),
        cmap=CMAP_E,
        norm=NORM_E,
        s=s_sat,
        transform=ccrs.PlateCarree(),
        zorder=zorder_sat,
    )

    # difference
    e_diff = ds_es.e.sel(ix_sat_int=ix, surf_refl="L") - ds_es[
        "ea_e_mean"
    ].sel(ix_sat_int=ix, surf_refl="L", ac_channel=setup["ac"])
    im = axes_row[3].scatter(
        ds_es.lon_sat.sel(ix_sat_int=ix),
        ds_es.lat_sat.sel(ix_sat_int=ix),
        c=e_diff,
        cmap=CMAP_D,
        norm=NORM_D,
        s=s_sat,
        transform=ccrs.PlateCarree(),
        zorder=zorder_sat,
    )

    # iqr of aircraft averaged to satellite
    im = axes_row[4].scatter(
        ds_es.lon_sat.sel(ix_sat_int=ix),
        ds_es.lat_sat.sel(ix_sat_int=ix),
        c=ds_es["ea_e_iqr"].sel(
            ix_sat_int=ix, surf_refl="L", ac_channel=setup["ac"]
        ),
        cmap=CMAP_S,
        norm=NORM_S,
        s=s_sat,
        transform=ccrs.PlateCarree(),
        zorder=zorder_sat,
    )

    # add scale bar in bottom left corner
    for ax in axes_row.flatten():
        add_scale(
            ax=ax,
            proj=CRS_PROJ,
            **SCALE_PARAMS[flight_id],
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


def read_data(flight_id):
    """
    Read data for map

    Returns
    -------
    ds_ea: xr.Dataset
        Aircraft emissivity
    ds_es: xr.Dataset
        Satellite emissivity
    """

    # read satellite footprints with aircraft resampled to satellite
    ds_es = read_airsat(flight_id=flight_id)

    # read aircraft emissivity at high resolution
    ds_ea = read_emissivity_aircraft(flight_id=flight_id)
    ds_ea = airborne_filter(ds_ea, drop_times=True)

    return ds_ea, ds_es


def add_scale(ax, width, step, proj, annotation_params=dict()):
    """
    Adds km-scale on cartopy map in the lower left corner. The scale can be
    modified in many ways.

    The scale bar is shown on the very top of the figure.

    (x0, y0): lower left corner of scale bar
    (y1, y1): upper left corner of scale bar
    x=(), y=(y0, y1): discrete bins of scale bar

    Parameters
    ----------
    ax: axis to add the scale on.
    width: scale width in km (e.g. 100 km).
    step: step of color scale in km (e.g. 25 km).
    proj: map projection.
    """

    assert (
        width / step % 2 == 0
    ), "Scale with divided by step must be an even number"

    # coordinates to m
    width = width * 1e3
    step = step * 1e3

    # get point on map coordinates that is 10% from the lower left corner
    xl0, xl1 = ax.get_xlim()
    yl0, yl1 = ax.get_ylim()
    dx = 0.02 * (xl1 - xl0)
    dy = 0.02 * (yl1 - yl0)
    x0 = xl0 + dx
    y0 = yl0 + dy

    height = 0.1 * width
    y1 = y0 + height
    x1 = x0 + width
    x = np.arange(x0, x1, step) + step / 2
    y = np.array([y0, y1])
    c = np.array([[0, 1] * int(len(x) / 2), [0, 1] * int(len(x) / 2)])

    ax.annotate(
        f"{int(width*1e-3)} km",
        xy=(x0, y1),
        ha="left",
        va="bottom",
        transform=proj,
        zorder=98,
        **annotation_params,
    )
    ax.pcolormesh(
        x, y, c, cmap="Greys", transform=proj, shading="nearest", zorder=99
    )


def diameter_to_scattersize(ax, diameter):
    """
    Converts diameter to the size used as input to plt.scatter(). The s
    parameter requires the size in points**2. This is calculated here. Use
    case is the visualization of circles on a map with a given diameter.

    See also the matplotlib documentation.
    s = marker size in points**2 (typographic points are 1/72 in.)

    Basic approach:
    map length (m) --> size in inches --> size in points --> size in points**2

    Caution: any rescaling of the figure leads to wrong point sizes.

    Note: use fig.canvas.draw() before plotting this

    Parameters
    ----------
    ax: geo axis object
    diameter: diameter in meters (or other, equal to the map unit)

    Returns
    -------
    s: diameter in unit points**2, which is input to scatter() function.
    """

    # transform diameter to matplotlib scatter size
    ppd = 72.0 / ax.figure.dpi
    trans = ax.transData.transform

    # check if diameter is a numpy array
    if isinstance(diameter, np.ndarray):
        s = (
            (
                trans((diameter, diameter))
                - trans((np.zeros(len(diameter)), np.zeros(len(diameter))))
            )
            * ppd
        )[:, 1]

    else:
        s = ((trans((diameter, diameter)) - trans((0, 0))) * ppd)[1]

    s = s**2

    return s


if __name__ == "__main__":
    main()
