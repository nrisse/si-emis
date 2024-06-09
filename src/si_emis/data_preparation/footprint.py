"""
Calculates footprint location in lat/lon/alt from aircraft position and viewing
angles and topography information.

The conversion is based on the description in the MiRAC paper.

Example:
    python -m data_preparation.footprint HALO-AC3 HALO

Check help:
    python -m data_preparation.footprint -h

Possible extension:
Add flag to indicate interpolated gps/ins values
"""

import argparse

import ac3airborne
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from lizard.ac3airlib import get_all_flights
from lizard.bitflags import write_flag_xarray
from lizard.readers.gps_ins import read_gps_ins

from si_emis.data_preparation.surface import (
    glacier_mask,
    land_ocean_shore_mask,
    sic_mask,
)
from si_emis.data_preparation.terrain import FootprintTerrain
from si_emis.data_preparation.transformation import FootprintLocation
from si_emis.writers.footprint import write_footprint

load_dotenv()
plt.ion()

META = ac3airborne.get_flight_segments()


def footprint_location(flight_id, viewing_angles=(0, -25)):
    """
    Computes footprint location over ocean and land, and incidence angle over
    flat surface for a given research flight and a given set of instrument
    viewing angles.

    Parameters
    -------
    flight_id: research flight id.
    viewing_angles: instrument viewing angles, for which footprint location
    will be calculated.
    """

    mission, platform, name = flight_id.split("_")
    flight = META[mission][platform][flight_id]
    ds_gps = read_gps_ins(flight_id)

    # create output dataset
    ds = xr.Dataset()
    ds.coords["time"] = pd.date_range(
        flight["takeoff"], flight["landing"], freq="1S"
    )
    ds.coords["view_ang"] = np.array(viewing_angles)

    # add gps variables
    ds["subac_lon"] = ds_gps["lon"]
    ds["subac_lat"] = ds_gps["lat"]
    ds["ac_alt"] = ds_gps["alt"]
    ds["ac_roll"] = ds_gps["roll"]
    ds["ac_pitch"] = ds_gps["pitch"]
    ds["ac_heading"] = ds_gps["heading"]

    # interpolate gps variables if needed
    ds = ds.interpolate_na("time", "linear")

    # drop remaining nan values, for example at start of flight
    ds = ds.sel(time=~ds.subac_lon.isnull())

    # add output variables
    ds["lon"] = (
        ("time", "view_ang"),
        np.full((len(ds.time), len(ds.view_ang)), fill_value=np.nan),
    )
    ds["lat"] = xr.zeros_like(ds["lon"])
    ds["surf_alt"] = xr.zeros_like(ds["lon"])
    ds["ac_zen"] = xr.zeros_like(ds["lon"])
    ds["ac_range"] = xr.zeros_like(ds["lon"])
    ds["land_mask"] = xr.zeros_like(ds["lon"], dtype="uint8")
    ds["ocean_mask"] = xr.zeros_like(ds["lon"], dtype="uint8")
    ds["shore_mask"] = xr.zeros_like(ds["lon"], dtype="uint8")
    ds["glacier_mask"] = xr.zeros_like(ds["lon"], dtype="uint8")
    ds["sic"] = xr.zeros_like(ds["lon"], dtype="uint8")
    ds["sea_ice_mask"] = xr.zeros_like(ds["lon"], dtype="uint8")

    for i, viewing_angle in enumerate(viewing_angles):
        # calculate footprint and incidence angle location over ocean
        fl = FootprintLocation(
            alpha=0,
            beta=viewing_angle,
            heading=ds["ac_heading"].values,
            pitch=ds["ac_pitch"].values,
            roll=ds["ac_roll"].values,
            lon=ds["subac_lon"].values,
            lat=ds["subac_lat"].values,
            h=ds["ac_alt"].values,
        )
        fl.get_footprint()

        ds["lon"][:, i] = fl.r_geod[:, 0, 0]
        ds["lat"][:, i] = fl.r_geod[:, 0, 1]
        ds["surf_alt"][:, i] = fl.r_geod[:, 0, 2]
        ds["ac_zen"][:, i] = fl.theta

        # add land sea mask (including shore class as +/- 150 m from coastline)
        land_mask, ocean_mask, shore_mask = land_ocean_shore_mask(
            ds["lon"][:, i], ds["lat"][:, i]
        )
        ds["land_mask"][:, i] = land_mask
        ds["ocean_mask"][:, i] = ocean_mask
        ds["shore_mask"][:, i] = shore_mask

        # footprint over land (= Svalbard only)
        mask = dict(time=ds["land_mask"][:, i] == 1)
        if mask["time"].sum() > 0:
            ft = FootprintTerrain(
                mission=mission,
                alpha=0,
                beta=viewing_angle,
                heading=ds["ac_heading"].sel(**mask).values,
                pitch=ds["ac_pitch"].sel(**mask).values,
                roll=ds["ac_roll"].sel(**mask).values,
                lon=ds["subac_lon"].sel(**mask).values,
                lat=ds["subac_lat"].sel(**mask).values,
                h=ds["ac_alt"].sel(**mask).values,
            )
            ft.get_footprint()

            # store lon/lat along line of sight with height from terrain model
            ds["lon"][mask["time"], i] = ft.los_geod_footprint[:, 0]
            ds["lat"][mask["time"], i] = ft.los_geod_footprint[:, 1]
            ds["surf_alt"][mask["time"], i] = ft.dtm_geod_footprint[:, 2]

        # compute distance between footprint and aircraft
        p0 = ccrs.Geocentric().transform_points(
            ccrs.Geodetic(),
            ds["lon"][:, i],
            ds["lat"][:, i],
            ds["surf_alt"][:, i],
        )

        p1 = ccrs.Geocentric().transform_points(
            ccrs.Geodetic(),
            ds["subac_lon"],
            ds["subac_lat"],
            ds["ac_alt"],
        )

        ds["ac_range"][:, i] = np.linalg.norm(p1 - p0, axis=1)

        # glacier mask
        ds["glacier_mask"][:, i] = glacier_mask(
            ds["lon"][:, i], ds["lat"][:, i]
        )

        # sea ice concentration and mask from AMSR2 product
        ds["sic"][:, i], ds["sea_ice_mask"][:, i] = sic_mask(
            ds["lon"][:, i], ds["lat"][:, i], flight_id
        )

    # write bit surface mask
    ds = write_flag_xarray(
        ds,
        bool_vars=[
            "ocean_mask",
            "land_mask",
            "shore_mask",
            "glacier_mask",
            "sea_ice_mask",
        ],
        bool_meanings=["ocean", "land", "shore", "glacier", "sea_ice"],
        bit_mask_name="surf_type",
        drop=True,
    )

    # add attributes
    ds["time"].attrs = dict(
        standard_name="time", long_name="time in seconds since epoch"
    )
    ds["time"].encoding = dict(
        units="seconds since 1970-01-01", calendar="standard"
    )

    ds["view_ang"].attrs = dict(
        standard_name="sensor_view_angle",
        long_name="view angle",
        description="off-nadir pointing angle",
        units="degree",
    )

    ds["lon"].attrs = dict(
        standard_name="longitude",
        long_name="longitude",
        description="longitude of footprint with terrain correction",
        units="degrees_east",
    )

    ds["lat"].attrs = dict(
        standard_name="latitude",
        long_name="latitude",
        description="latitude of footprint with terrain correction",
        units="degrees_north",
    )

    ds["surf_alt"].attrs = dict(
        standard_name="surface_altitude",
        long_name="surface altitude",
        description="surface altitude of footprint",
        comment="surface altitude is interpolated from the digital terrain "
        "model Terrengmodell Svalbard (S0 Terrengmodell) of the Norwegian "
        "Polar Institute available at "
        "https://doi.org/10.21334/npolar.2014.dce53a47",
        units="m",
    )

    ds["subac_lon"].attrs = dict(
        standard_name="longitude",
        long_name="sub-aircraft longitude",
        description="sub-aircraft longitude",
        units="degrees_east",
    )

    ds["subac_lat"].attrs = dict(
        standard_name="latitude",
        long_name="sub-aircraft latitude",
        description="sub-aircraft latitude",
        units="degrees_north",
    )

    ds["ac_alt"].attrs = dict(
        standard_name="altitude",
        long_name="aircraft altitude",
        description="aircraft altitude",
        units="m",
    )

    ds["ac_heading"].attrs = dict(
        standard_name="heading",
        long_name="aircraft heading",
        description="aircraft heading (0=north, 90=east, 180/-180=south, -90=west)",
        units="degree",
    )

    ds["ac_pitch"].attrs = dict(
        standard_name="pitch",
        long_name="aircraft pitch",
        description="aircraft pitch (positive for nose above horizon)",
        units="degree",
    )

    ds["ac_roll"].attrs = dict(
        standard_name="roll",
        long_name="roll",
        description="aircraft roll (positive for right wing down)",
        units="degree",
    )

    ds["ac_zen"].attrs = dict(
        standard_name="aircraft_zenith_angle",
        long_name="aircraft zenith angle",
        description="aircraft zenith angle at footprint",
        units="degree",
    )

    ds["ac_range"].attrs = dict(
        standard_name="aircraft_range",
        long_name="aircraft range",
        description="line of sight distance between aircraft and footprint",
        units="m",
    )

    ds["surf_type"].attrs.update(
        dict(
            standard_name="surface_type",
            long_name="surface type",
            description="surface type at footprint",
        )
    )

    ds["sic"].attrs = dict(
        standard_name="sea_ice_mask",
        long_name="sea ice mask",
        description="Sea ice mask at footprint location extracted"
        " from asi-AMSR2 n6250 v5.4 by University of Bremen",
    )

    ds.attrs["source"] = (
        "See Risse et al. (2023) for a collection of references for this "
        "data. "
    )

    # write dataset to file
    write_footprint(ds, flight_id)

    print(f"Finished {flight_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mission",
        help="mission name",
        type=str,
        choices=["ACLOUD", "AFLUX", "MOSAiC-ACA", "HALO-AC3", "HAMAG"],
        required=False,
    )
    parser.add_argument(
        "--platform",
        help="platform name",
        type=str,
        choices=["P5", "HALO", "P6"],
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

    for i, f in enumerate(flight_ids):
        print(f)

        m, p, n = f.split("_")

        if p == "HALO":
            footprint_location(flight_id=f, viewing_angles=[0])
        else:
            footprint_location(flight_id=f, viewing_angles=[0, -25])
