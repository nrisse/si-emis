"""
Read PAMTRA simulation
"""

import os

import xarray as xr
from dotenv import load_dotenv

load_dotenv()


def read_pamtra_simulation(filename):
    """
    Reads PAMTRA simulation from dedicated location.

    Parameters
    ----------
    ds: dataset with PAMTRA simulation
    filename: output filename
    """

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/pamtra_simulation",
            filename,
        )
    )

    # pamtra setting to dict
    ds.attrs["pyPamtra_settings"] = {
        x.split(": ")[0]: x.split(": ")[1]
        for x in ds.attrs["pyPamtra_settings"].split(", ")
    }

    return ds
