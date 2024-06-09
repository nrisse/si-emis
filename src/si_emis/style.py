"""
Definition of matplotlib style, campaign and instrumentation info, and colors.
"""

import cmcrameri.cm as cmc
import matplotlib as mpl
import numpy as np

# matplotlib style
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["figure.figsize"] = [5, 4]
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"

# field campaign time periods
missions = {
    "ACLOUD": {
        "start": np.datetime64("2017-05-23"),
        "end": np.datetime64("2017-06-26"),
    },
    "AFLUX": {
        "start": np.datetime64("2019-03-19"),
        "end": np.datetime64("2019-04-11"),
    },
    "MOSAiC-ACA": {
        "start": np.datetime64("2020-08-30"),
        "end": np.datetime64("2020-09-13"),
    },
    "HALO-AC3": {
        "start": np.datetime64("2022-03-12"),
        "end": np.datetime64("2022-04-12"),
    },
}

# instrumentation
sensors = {  # sensors for each campaign
    "ACLOUD": {"P5": dict(sensor1="MiRAC-A", sensor2="MiRAC-P")},
    "AFLUX": {"P5": dict(sensor1="MiRAC-A", sensor2="MiRAC-P")},
    "MOSAiC-ACA": {"P5": dict(sensor1="MiRAC-A", sensor2="HATPRO")},
    "HALO-AC3": {
        "P5": dict(sensor1="MiRAC-A", sensor2="HATPRO"),
        "HALO": dict(sensor1="HAMP", sensor2=None),
    },
}

viewing_angles = {  # channel viewing angles for each campaign by sensor order
    "ACLOUD": {"P5": [-25, 0, 0, 0, 0, 0, 0, 0, 0]},
    "AFLUX": {"P5": [-25, 0, 0, 0, 0, 0, 0, 0, 0]},
    "MOSAiC-ACA": {"P5": [-25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    "HALO-AC3": {
        "P5": [-25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "HALO": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    },
}

# mirac viewing angles and polarization as text
mirac_pol_text = {
    1: "H-pol, 25°",
    2: "0°",
    3: "0°",
    4: "0°",
    5: "0°",
    6: "0°",
    7: "0°",
    8: "0°",
    9: "0°",
}

# research flight colors
flight_colors = {
    "ACLOUD_P5_RF23": cmc.oleron(np.linspace(0, 1, 6))[0],
    "ACLOUD_P5_RF25": cmc.oleron(np.linspace(0, 1, 6))[2],
    "AFLUX_P5_RF08": cmc.oleron(np.linspace(0, 1, 6))[3],
    "AFLUX_P5_RF14": cmc.oleron(np.linspace(0, 1, 6))[4],
    "AFLUX_P5_RF15": cmc.oleron(np.linspace(0, 1, 6))[5],
}

# radiometers: MiRAC-A + MiRAC-P
mirac_cols = {
    1: cmc.batlow(np.linspace(0, 0.8, 4))[3],  # close to channel 9
    2: cmc.batlow(np.linspace(0, 0.8, 6))[5],  # close to channel 3
    3: cmc.batlow(np.linspace(0, 0.8, 6))[4],
    4: cmc.batlow(np.linspace(0, 0.8, 6))[3],
    5: cmc.batlow(np.linspace(0, 0.8, 6))[2],
    6: cmc.batlow(np.linspace(0, 0.8, 6))[1],  # close to channel 7
    7: cmc.batlow(np.linspace(0, 0.8, 6))[0],
    8: cmc.batlow(np.linspace(0, 0.8, 4))[1],  # close to color of channel 7
    9: cmc.batlow(np.linspace(0, 0.8, 4))[2],  # close to channel 8
}

# satellite colors (brown for Metop, blue for NOAA and SNPP, green for DMSP)
platform_colors = {
    "Aqua": {"color": "rosybrown", "linestyle": "-"},
    "Metop-A": {"color": "peru", "linestyle": "-"},
    "Metop-B": {"color": "sandybrown", "linestyle": "-"},
    "Metop-C": {"color": "saddlebrown", "linestyle": "-"},
    "NOAA-15": {"color": "dodgerblue", "linestyle": "-"},
    "NOAA-18": {"color": "deepskyblue", "linestyle": "-"},
    "NOAA-19": {"color": "aqua", "linestyle": "-"},
    "NOAA-20": {"color": "teal", "linestyle": "-"},
    "SNPP": {"color": "navy", "linestyle": "-"},
    "GCOM-W": {"color": "violet", "linestyle": "--"},
    "DMSP-F15": {"color": "forestgreen", "linestyle": "--"},
    "DMSP-F16": {"color": "limegreen", "linestyle": "-."},
    "DMSP-F17": {"color": "mediumseagreen", "linestyle": "-."},
    "DMSP-F18": {"color": "lime", "linestyle": "-."},
    "Polar 5": {"color": "darkgray", "linestyle": "-"},
}

# sensor colors (similar to platform colors but scientific colors)
sensor_colors = {
    "MHS": cmc.batlowK(np.linspace(0, 0.8, 5))[1],
    "ATMS": cmc.batlowK(np.linspace(0, 0.8, 5))[2],
    "SSMIS": cmc.batlowK(np.linspace(0, 0.8, 5))[3],
    "AMSR2": cmc.batlowK(np.linspace(0, 0.8, 5))[4],
    "MiRAC": cmc.batlowK(np.linspace(0, 0.8, 5))[0],
}

pol_markers = {
    0: {"label": "VH", "marker": "o"},
    1: {"label": "QV", "marker": "d"},
    2: {"label": "QH", "marker": "D"},
    3: {"label": "V", "marker": "v"},
    4: {"label": "H", "marker": "h"},
    5: {"label": "S3", "marker": "o"},
    6: {"label": "S4", "marker": "o"},
    7: {"label": "xVH", "marker": "o"},
}

# k-means cluster colors
kmeans_cmap = cmc.batlow(np.linspace(0, 1, 4))
kmeans_colors = {
    1: kmeans_cmap[0],
    2: kmeans_cmap[1],
    3: kmeans_cmap[2],
    4: kmeans_cmap[3],
}
