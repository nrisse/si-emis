"""
Read atmospheric profile that was constructed for each research flight.
"""

import os

import xarray as xr
from dotenv import load_dotenv

load_dotenv()


def read_profile(flight_id):
    """
    Read in-situ atmospheric profile that was constructed for each research
    flight.

    Parameters
    ----------
    flight_id: ac3airborne flight id

    Returns
    -------
    ds: xarray.Dataset of atmospheric profile
    """

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/atmospheric_profile",
            f"profile_{flight_id}.nc",
        )
    )

    return ds
