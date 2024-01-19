"""
Write PAMTRA simulation to file
"""

import os

from dotenv import load_dotenv

load_dotenv()


def write_pamtra_simulation(ds, filename):
    """
    Writes PAMTRA simulation to dedicated location.

    Parameters
    ----------
    ds: dataset with PAMTRA simulation
    filename: output filename
    """

    file = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_emissivity/pamtra_simulation",
        filename,
    )

    ds.to_netcdf(file)

    print(f"Written file: {file}")
