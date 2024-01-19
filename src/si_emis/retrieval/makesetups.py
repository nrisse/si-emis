"""
This script creates yaml setup files for pamtra simulations. Each setup file
contains one research flight. The pamtra simulation script requires the setup
files as input argument. Each setup script contains a dictionary of setups with
a unique id number. This number is also written into the filename to assign
a certain output file to its setting. However, the setting will be also written
into the global attributes of the output file.

Setup rules (30 setups in total, from 00 to 29):
  - natural simulation: both polarizations, ocean as specular, emissivity will
    not be used, no simulations with offsets (--> only land/sea ice in S/L)
    --> 2 simulations (00=lamb and 01=spec)
  - idealized simulation: only one polarization (because their emissivity and
    tb is equal), emissivitiy 0 or 1, offsets vary, surface reflections vary
    --> 2*2*7=28 simulations (02-29, 02-15: lamb, 16-29: spec, 09-15 and 
        23-29: e=1, 02-08 and 16-22: e=0, 02 and 09 and 16 and 23: no offset
        and then varying offsets until 08 and 15 and 22 and 29)
    --> for emissivity retrieval, combine these element-wise (e.g. 02 and 09): 
        lamb: e=0 02-08 with e=1 09-15 (no offset and the six offsets)
        spec: e=0 16-22 with e=1 23-29 (no offset and the six offsets)

Example of a setup entry:
'00':
  general:
    flight_id: ACLOUD_P5_RF23
    outfile: pamtra_ACLOUD_P5_RF23_00.nc
  nmlSet:
    creator: Nils Risse
    active: false
    passive: true
    outpol: VH
    add_obs_height_to_layer: true
    gas_mod: R98
    emissivity: 0
  offset:
    groundtemp: 0
    temp: 0
    relhum: 0
  surface:
    kind: natural
    ocean_model: TESSEM2
    ocean_refl: S
    land_model: TELSEM2
    land_refl: L
    sea_ice_model: TELSEM2
    sea_ice_refl: L
"""

from datetime import datetime as dt

import yaml

flight_ids = [
    "ACLOUD_P5_RF23",
    "ACLOUD_P5_RF25",
    "AFLUX_P5_RF08",
    "AFLUX_P5_RF14",
    "AFLUX_P5_RF15",
]
times = {
    "ACLOUD_P5_RF23": [["2017-06-25 11:35:00", "2017-06-25 16:41:30"]],
    "ACLOUD_P5_RF25": [
        ["2017-06-26 13:10:00", "2017-06-26 13:23:30"],
        ["2017-06-26 13:43:00", "2017-06-26 13:57:00"],
    ],
    "AFLUX_P5_RF07": [["2019-03-30 15:18:00", "2019-03-30 15:21:00"]],
    "AFLUX_P5_RF08": [["2019-03-31 10:20:00", "2019-03-31 11:51:00"]],
    "AFLUX_P5_RF13": [
        ["2019-04-07 07:43:00", "2019-04-07 07:45:00"],
        ["2019-04-07 11:37:00", "2019-04-07 11:41:00"],
    ],
    "AFLUX_P5_RF14": [["2019-04-08 10:14:00", "2019-04-08 10:57:00"]],
    "AFLUX_P5_RF15": [["2019-04-11 10:29:00", "2019-04-11 11:58:00"]],
    "HALO-AC3_P5_RF07": [["2022-03-29 12:55:30", "2022-03-29 16:03:40"]],
    "HALO-AC3_HALO_RF10": [["2022-03-29 14:12:00", "2022-03-29 14:40:30"]],
}


def read_setups(file):
    """
    Reads PAMTRA setup from yaml file.

    Parameters
    ----------
    file: filename of setup file

    Returns
    -------
    setup: dict with setup keys
    """

    with open(file, "r") as setupfile:
        setup = yaml.safe_load(setupfile)

    return setup


def write_setups_aircraft():
    """
    Writes setup files for aircraft simulations

    Main difference to satellite simulations:
    - adds observation height to levels
    - calculates several offsets
    - uses specific time slices

    """

    refls = ["L", "S"]
    kinds = ["natural", "idealized"]
    emissivities = [0, 1]

    setups = dict()
    for flight_id in flight_ids:
        i = 0

        mission, platform, name = flight_id.split("_")

        # this ensures that AFLUX uses a higher groundtemp offset than ACLOUD
        if mission == "ACLOUD":
            offsets = [  # (groundtemp, temp, relhum)
                (0, 0, 0),
                (-3, 0, 0),  # this is different to AFLUX
                (3, 0, 0),  # this is different to AFLUX
                (0, -2, 0),
                (0, 2, 0),
                (0, 0, -5),
                (0, 0, 5),
            ]

        elif mission in ["AFLUX", "HALO-AC3"]:
            offsets = [  # (groundtemp, temp, relhum)
                (0, 0, 0),
                (-8, 0, 0),  # this is different to ACLOUD
                (8, 0, 0),  # this is different to ACLOUD
                (0, -2, 0),
                (0, 2, 0),
                (0, 0, -5),
                (0, 0, 5),
            ]

        else:
            raise ValueError(
                f"Unknown mission for offset definition {mission}"
            )

        for kind in kinds:
            for refl in refls:
                if kind == "natural":
                    setups[str(i).zfill(2)] = {
                        "general": {
                            "flight_id": flight_id,
                            "times": [
                                [
                                    dt.strptime(t0, "%Y-%m-%d %H:%M:%S"),
                                    dt.strptime(t1, "%Y-%m-%d %H:%M:%S"),
                                ]
                                for t0, t1 in times[flight_id]
                            ],
                            "outfile": f"pamtra_{flight_id}_"
                            f"{str(i).zfill(2)}.nc",
                        },
                        "nmlSet": {
                            "creator": "Nils Risse",
                            "active": False,
                            "passive": True,
                            "outpol": "VH",
                            "add_obs_height_to_layer": True,
                            "gas_mod": "R98",
                            "emissivity": 0,
                        },
                        "offset": {"groundtemp": 0, "temp": 0, "relhum": 0},
                        "surface": {
                            "kind": kind,
                            "ocean_model": "TESSEM2",
                            "ocean_refl": "S",
                            "land_model": "TELSEM2",
                            "land_refl": refl,
                            "sea_ice_model": "TELSEM2",
                            "sea_ice_refl": refl,
                        },
                    }
                    i += 1
                elif kind == "idealized":
                    for emissivity in emissivities:
                        for offset in offsets:
                            setups[str(i).zfill(2)] = {
                                "general": {
                                    "flight_id": flight_id,
                                    "times": [
                                        [
                                            dt.strptime(
                                                t0, "%Y-%m-%d %H:%M:%S"
                                            ),
                                            dt.strptime(
                                                t1, "%Y-%m-%d %H:%M:%S"
                                            ),
                                        ]
                                        for t0, t1 in times[flight_id]
                                    ],
                                    "outfile": f"pamtra_{flight_id}_"
                                    f"{str(i).zfill(2)}.nc",
                                },
                                "nmlSet": {
                                    "creator": "Nils Risse",
                                    "active": False,
                                    "passive": True,
                                    "outpol": "V",
                                    "add_obs_height_to_layer": True,
                                    "gas_mod": "R98",
                                    "emissivity": emissivity,
                                },
                                "offset": {
                                    "groundtemp": offset[0],
                                    "temp": offset[1],
                                    "relhum": offset[2],
                                },
                                "surface": {
                                    "kind": kind,
                                    "ocean_model": "TESSEM2",
                                    "ocean_refl": refl,
                                    "land_model": "TELSEM2",
                                    "land_refl": refl,
                                    "sea_ice_model": "TELSEM2",
                                    "sea_ice_refl": refl,
                                },
                            }
                            i += 1

        outfile = f"./retrieval/pamtra/setups/setup_{flight_id}.yaml"
        with open(outfile, "w") as f:
            yaml.dump(setups, f, sort_keys=False)


def write_setups_satellite():
    """
    Creates setup files for satellites.

    Main difference to aircraft:
    - observation height is not added to levels
    - no offsets are applied
    """

    refls = ["L", "S"]
    kinds = ["natural", "idealized"]
    emissivities = [0, 1]

    setups = dict()
    for flight_id in flight_ids:
        i = 0
        for kind in kinds:
            for refl in refls:
                if kind == "natural":
                    setups[str(i).zfill(2)] = {
                        "general": {
                            "flight_id": flight_id,
                            "outfile": f"pamtra_sat_{flight_id}_"
                            f"{str(i).zfill(2)}.nc",
                        },
                        "nmlSet": {
                            "creator": "Nils Risse",
                            "active": False,
                            "passive": True,
                            "outpol": "VH",
                            "add_obs_height_to_layer": False,
                            "gas_mod": "R98",
                            "emissivity": 0,
                        },
                        "surface": {
                            "kind": kind,
                            "ocean_model": "TESSEM2",
                            "ocean_refl": "S",
                            "land_model": "TELSEM2",
                            "land_refl": refl,
                            "sea_ice_model": "TELSEM2",
                            "sea_ice_refl": refl,
                        },
                    }
                    i += 1
                elif kind == "idealized":
                    for emissivity in emissivities:
                        setups[str(i).zfill(2)] = {
                            "general": {
                                "flight_id": flight_id,
                                "outfile": f"pamtra_sat_{flight_id}_"
                                f"{str(i).zfill(2)}.nc",
                            },
                            "nmlSet": {
                                "creator": "Nils Risse",
                                "active": False,
                                "passive": True,
                                "outpol": "V",
                                "add_obs_height_to_layer": False,
                                "gas_mod": "R98",
                                "emissivity": emissivity,
                            },
                            "surface": {
                                "kind": kind,
                                "ocean_model": "TESSEM2",
                                "ocean_refl": refl,
                                "land_model": "TELSEM2",
                                "land_refl": refl,
                                "sea_ice_model": "TELSEM2",
                                "sea_ice_refl": refl,
                            },
                        }
                        i += 1

        outfile = f"./retrieval/pamtra/setups/setup_sat_{flight_id}.yaml"
        with open(outfile, "w") as f:
            yaml.dump(setups, f, sort_keys=False)


if __name__ == "__main__":
    write_setups_aircraft()
    write_setups_satellite()
