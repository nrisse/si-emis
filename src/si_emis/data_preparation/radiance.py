"""
Combines footprint dataset with observed microwave and infrared brightness
temperatures (converted to surface temperature). Instruments are combined to
one output dataset. Slanted view of MiRAC-A is matched with nadir-viewing
infrared radiometer. Time shifts between MiRAC (both MiRAC-A and MiRAC-P) and
KT-19 are applied. These were determined from lagged correlations between
MiRAC-P at 243 GHz and KT-19, as these view both at nadir. MiRAC-A and MiRAC-P
measure at the same time.

Note that this script does not use MiRAC-A product from the PANGAEA website but
a version that interpolates the data at ms-resolution to s-resolution. This
data is provided in the published emissivity dataset. Generally, the PANGAEA
MiRAC-A files are similar to this and can be used as well but contain slightly
less samples.

Example:
    python -m data_preparation.radiance ACLOUD P5

Check help:
    python -m data_preparation.radiance -h
"""

import argparse

import numpy as np
import xarray as xr
from lizard.ac3airlib import (
    get_all_flights,
    get_mirac_a_flight_ids,
    get_mirac_p_hatpro_flight_ids,
)
from lizard.readers.band_pass import read_band_pass, read_band_pass_combination
from lizard.readers.bbr import read_bbr
from lizard.readers.footprint import read_footprint
from lizard.readers.halo_kt19 import read_halo_kt19
from lizard.readers.hamp import read_hamp
from lizard.readers.mirac_a_tb_l1 import read_mirac_a_tb_l1

from si_emis.data_preparation.matching import Matching
from si_emis.readers.airborne_tb import read_from_intake
from si_emis.style import sensors, viewing_angles
from si_emis.writers.brightness_temperature import write_tb

# time shift in seconds applied to MiRAC-A and MiRAC-P to match with KT-19
# these values come from the figure created with timeshifts.py
# note that any modifications here do not affect the PAMTRA simulation, because
# KT-19 and GPS are fixed in time
TIME_SHIFTS = {
    "ACLOUD_P5_RF23": 1,
    "ACLOUD_P5_RF25": 2,
    "AFLUX_P5_RF07": 2,
    "AFLUX_P5_RF08": 3,
    "AFLUX_P5_RF13": 3,
    "AFLUX_P5_RF14": 4,
    "AFLUX_P5_RF15": 5,
    "HALO-AC3_P5_RF07": 0,
    "HALO-AC3_HAMP_RF10": 0,
}

# infrared emissivity to convert KT-19 TB to surface temperature
IR_EMISSIVITY = 0.995


def radiance2footprint(flight_id):
    """
    Adds brightness temperature observations to footprint dataset for a given
    flight_id. The sensors and viewing angles of each channel are defined
    outside of this function.

    Future extension: Make compatible with MOSAiC-ACA and HALO-AC3 (Polar 5 and
    HALO)

    Dimensions of the footprint dataset: (time, view_ang)
    Dimensions of the tb dataset: (time, channel)
    """

    flight_ids_mirac_a = get_mirac_a_flight_ids()
    flight_ids_mirac_p_hatpro = get_mirac_p_hatpro_flight_ids()

    mission, platform, name = flight_id.split("_")

    sensor1 = sensors[mission][platform]["sensor1"]
    sensor2 = sensors[mission][platform]["sensor2"]
    channel_view_ang = viewing_angles[mission][platform]

    # get footprint dataset as basis
    ds = read_footprint(flight_id)

    # add channel information
    if sensor1 is not None and sensor2 is not None:
        ds_bp = read_band_pass_combination(sensor1, sensor2)
    else:
        ds_bp = read_band_pass(sensor1)
    ds.coords["channel"] = ds_bp.channel
    vars_to_add = [
        "center_freq",
        "n_if_offsets",
        "if_offset_1",
        "if_offset_2",
        "bandwidth",
        "polarization",
    ]
    for v in vars_to_add:
        ds[v] = ds_bp[v]

    # add viewing angle
    ds["channel_view_ang"] = ("channel", channel_view_ang)

    # add tb data
    if platform == "P5":
        # read tb data
        if flight_id in flight_ids_mirac_p_hatpro:
            ds_tb_hat = read_from_intake(flight_id)
            ds_tb_hat["channel"] = ds_tb_hat["channel"] + 1
        else:
            ds_tb_hat = None

        if flight_id in flight_ids_mirac_a:
            ds_tb_89 = read_mirac_a_tb_l1(flight_id)
            ds_tb_89 = ds_tb_89.expand_dims({"channel": [1]}, axis=-1)
        else:
            ds_tb_89 = None

        if ds_tb_89 is None and ds_tb_hat is not None:
            da_tb = ds_tb_hat
        elif ds_tb_89 is not None and ds_tb_hat is None:
            da_tb = ds_tb_89
        else:
            da_tb = xr.merge([ds_tb_89.tb, ds_tb_hat.tb])

        ds["tb"] = da_tb.tb

        # apply time shift to tb data on regular 1s time step
        shift = TIME_SHIFTS.get(flight_id)
        if shift:
            ds["tb"] = ds.tb.shift({"time": shift}, fill_value=np.nan)
            shift_comment = (
                f"a time shift of {shift}s is applied based on"
                f"lagged correlations of the 243 GHz TB with KT-19"
            )
        else:
            shift_comment = f"no time shift is applied"

        ds["tb"].attrs = dict(
            standard_name="brightness_temperature",
            long_name="brightness temperature",
            description=f"brightness temperature measured by {sensor1} "
            f"and {sensor2}",
            comment=shift_comment,
            units="K",
        )

        # add kt19 for nadir (natural) and matched with -25 degrees angle
        da_ts = read_bbr(flight_id).KT19

        # apply infrared emissivity
        da_ts = gray_body_temp(t_blackbody=da_ts, emissivity=IR_EMISSIVITY)

        match = Matching.from_flight_id(flight_id, from_angle=0, to_angle=-25)
        match.match(threshold=200, temporally=True)
        da_ts_from_angle, match.ds_fpr = xr.align(
            da_ts, match.ds_fpr, join="outer"
        )
        da_ts_to_angle = da_ts_from_angle.sel(
            time=match.ds_fpr.from_time
        ).where(match.ds_fpr.from_valid.values)
        da_ts_to_angle["time"] = da_ts_from_angle.time
        ds["ts"] = xr.concat(
            [da_ts_from_angle, da_ts_to_angle], dim="view_ang"
        ).T

        ds["ts"].attrs = dict(
            standard_name="surface_temperature",
            long_name="surface temperature",
            description="Surface temperature derived from the infrared (IR) "
            f"brightness temperature (assuming IR emissivity of "
            f"{IR_EMISSIVITY}) measured by "
            "Heitronics KT19.85II at nadir and matched with other"
            "viewing angles at the surface. Only valid under "
            "clear-sky conditions",
            units="K",
        )

        # remove viewing angle as dimension and make channel as variable
        for v in list(ds):
            if "view_ang" in ds[v].dims:
                ds[v] = ds[v].sel(view_ang=ds.channel_view_ang)
        ds = ds.drop("view_ang")
        ds = ds.rename({"channel_view_ang": "view_ang"})

    elif platform == "HALO":
        # this is adapted from polar 5. the main difference is that we only
        # have the 0 degrees viewing angle on halo
        # read tb data
        ds_tb_hamp = read_hamp(flight_id)

        # add to footprint dataset
        ds["tb"] = ds_tb_hamp.tb

        # apply time shift to tb data on regular 1s time step
        shift = TIME_SHIFTS.get(flight_id)
        if shift:
            ds["tb"] = ds.tb.shift({"time": shift}, fill_value=np.nan)
            shift_comment = (
                f"A time shift of {shift}s is applied based on"
                f"lagged correlations of the 90 GHz TB with KT-19"
            )
        else:
            shift_comment = f"No time shift is applied"

        ds["tb"].attrs = dict(
            standard_name="brightness_temperature",
            long_name="brightness temperature",
            description=f"Brightness temperature measured by {sensor1}. {shift_comment}",
            units="K",
        )

        # add kt19 for nadir (natural) and matched with -25 degrees angle
        da_ts = read_halo_kt19(flight_id).KT19
        da_ts = da_ts.expand_dims("view_ang", axis=1)

        # apply infrared emissivity
        ds["ts"] = gray_body_temp(t_blackbody=da_ts, emissivity=IR_EMISSIVITY)

        ds["ts"].attrs = dict(
            standard_name="surface_temperature",
            long_name="surface temperature",
            description="surface temperature derived from the infrared (IR) "
            f"brightness temperature (assuming IR emissivity of "
            f"{IR_EMISSIVITY}) measured by "
            "Heitronics KT19.85II at nadir and matched with other"
            "viewing angles at the surface. Only valid under "
            "clear-sky conditions",
            units="K",
        )

        # remove viewing angle as dimension and make channel as variable
        for v in list(ds):
            if "view_ang" in ds[v].dims:
                ds[v] = ds[v].sel(view_ang=ds.channel_view_ang)
        ds = ds.drop("view_ang")
        ds = ds.rename({"channel_view_ang": "view_ang"})

    # add attributes
    ds["time"].attrs = dict(
        standard_name="time", long_name="time in seconds since epoch"
    )
    ds["time"].encoding = dict(
        units="seconds since 1970-01-01", calendar="standard"
    )

    ds["channel"].attrs = dict(
        standard_name="channel",
        long_name="channel",
        description="microwave radiometer channel number",
    )

    ds["view_ang"].attrs = dict(
        standard_name="viewing_angle",
        long_name="viewing angle",
        description="viewing angle of the radiometer channel",
        units="degrees",
    )

    ds.attrs = dict(
        title="Combined microwave and infrared brightness temperature dataset"
        f"for the {platform} aircraft",
        history="used pre-calculated footprint locations; loaded TB from "
        "MiRAC-P from PANGAEA; loaded calibrated TB from MiRAC-A; "
        "loaded KT-19 data from PANGAEA; Matched KT-19 to slanted "
        "view of MiRAC-A;"
        "applied time "
        "shift from lagged correlation under "
        "clear-sky"
        "if possible",
        author="Nils Risse",
        contact="n.risse@uni-koeln.de",
        created=str(np.datetime64("now")),
    )

    write_tb(ds, flight_id)


def gray_body_temp(t_blackbody, emissivity):
    """
    Convert blackbody temperature to gray body temperature when measured with
    thermal infrared radiometer.
    """

    t_graybody = t_blackbody / emissivity

    return t_graybody


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mission",
        help="mission name",
        type=str,
        choices=["ACLOUD", "AFLUX", "MOSAiC-ACA", "HALO-AC3"],
        required=False,
    )
    parser.add_argument(
        "--platform",
        help="platform name",
        type=str,
        choices=["P5", "HALO"],
        required=False,
    )

    args = parser.parse_args()

    if args.mission is None:
        flight_ids = [
            "ACLOUD_P5_RF23",
            "ACLOUD_P5_RF25",
            "AFLUX_P5_RF08",
            "AFLUX_P5_RF14",
            "AFLUX_P5_RF15",
        ]

    else:
        flight_ids = get_all_flights(
            args.mission,
            args.platform,
        )

    for f in flight_ids:
        print(f)
        radiance2footprint(flight_id=f)
