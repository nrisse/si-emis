"""
Reads airborne emissivity resampled to satellite footprints
"""

import os

import xarray as xr


def read_airsat(flight_id):
    """
    Reads airborne emissivity resampled to satellite footprints

    Parameters
    ----------
    flight_id : str
        Flight ID

    Returns
    -------
    ds : xarray.Dataset
        Dataset with airborne emissivity resampled to satellite footprints
    """

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/airsat",
            f"{flight_id}_airsat.nc",
        )
    )

    return ds
