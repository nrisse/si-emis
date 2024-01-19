"""
Write file with emissivity and additionally data to file.
"""

import os

import ac3airborne
from dotenv import load_dotenv

load_dotenv()

META = ac3airborne.get_flight_segments()


def write_emissivity_aircraft(ds, flight_id):
    """
    Write emissivity dataset file.

    Parameters
    ----------
    ds: emissivity dataset
    flight_id: research flight id from ac3airborne.
    """

    mission, platform, name = flight_id.split("_")
    date = META[mission][platform][flight_id]["date"].strftime("%Y%m%d")

    ds.to_netcdf(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/emissivity",
            f"emissivity_{flight_id}_{date}.nc",
        )
    )


def write_emissivity_satellite(ds, flight_id):
    """
    Write emissivity dataset file for satellite overflights.

    Parameters
    ----------
    ds: emissivity dataset
    flight_id: research flight id from ac3airborne.
    """

    mission, platform, name = flight_id.split("_")
    date = META[mission][platform][flight_id]["date"].strftime("%Y%m%d")

    ds.to_netcdf(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/emissivity",
            f"emissivity_sat_{flight_id}_{date}.nc",
        )
    )
