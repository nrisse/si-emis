"""
Reads ESA CMEMS sea ice temperature. The daily data is retrieved for the ACLOUD
and AFLUX period (each in one file)
"""

import os

import pandas as pd
import xarray as xr
from dotenv import load_dotenv

load_dotenv()


def read_esa_cmems(date, roi=None):
    """
    Reads ESA CMEMS sea ice temperature for specific date

    Parameters
    ------
    date: this date will be loaded.
    roi: specify region of interest. (lon0, lon1, lat0, lat1)

    Returns
    -------
    ds: cmems sea ice temperature dataset
    """

    date_str = pd.Timestamp(date).strftime("%Y%m%d")

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sat/ne23",
            f"{date_str}120000-DMI-L4_GHRSST-STskin-DMI_OI-ARC_IST-v02.0-fv01.0.nc",
        )
    )

    ds = ds.isel(time=0)

    if roi is not None:
        lon0, lon1, lat0, lat1 = roi

        ix = (
            (ds.lon >= lon0)
            & (ds.lon <= lon1)
            & (ds.lat >= lat0)
            & (ds.lat <= lat1)
        )

        ds = ds.sel(lon=ix.any("lat"), lat=ix.any("lon"))

    return ds
