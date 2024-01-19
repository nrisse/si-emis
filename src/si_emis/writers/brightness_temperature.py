"""
Write file with brightness temperatures based on footprint positions and
TB time series.
"""

import os

import ac3airborne
from dotenv import load_dotenv

load_dotenv()

META = ac3airborne.get_flight_segments()


def write_tb(ds, flight_id):
    """
    Write brightness temperature dataset file.

    Parameters
    ----------
    ds: brightness temperature dataset
    flight_id: research flight id from ac3airborne.
    """

    mission, platform, name = flight_id.split("_")
    date = META[mission][platform][flight_id]["date"].strftime("%Y%m%d")

    ds.to_netcdf(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/brightness_temperature",
            f"tb_{flight_id}_{date}.nc",
        )
    )
