"""
Write file with footprint location. These files are combined afterwards with
the brightness temperatures and the basis for the final emissivity product.
"""

import os

import ac3airborne
from dotenv import load_dotenv
from lizard.ac3airlib import get_all_flights
from lizard.readers.footprint import read_footprint
from lizard.writers.xr2shp import xarray2shapefile

load_dotenv()

META = ac3airborne.get_flight_segments()


def write_footprint(ds, flight_id):
    """
    Write footprint dataset file.

    Parameters
    ----------
    ds: footprint dataset
    flight_id: research flight id from ac3airborne.
    """

    mission, platform, name = flight_id.split("_")
    date = META[mission][platform][flight_id]["date"].strftime("%Y%m%d")

    ds.to_netcdf(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/footprint",
            f"footprint_{flight_id}_{date}.nc",
        )
    )


def to_shape():
    """
    Converts footprint nc files to shp for use in Google Earth, GIS, etc.
    """

    flight_ids = get_all_flights(
        ["ACLOUD", "AFLUX", "HALO-AC3", "MOSAiC-ACA"], "P5"
    )

    for flight_id in flight_ids:
        ds = read_footprint(flight_id)

        mission, platform, name = flight_id.split("_")
        date = META[mission][platform][flight_id]["date"].strftime("%Y%m%d")

        ds = read_footprint(flight_id)
        xarray2shapefile(
            ds,
            os.path.join(
                os.environ["PATH_SEC"],
                "data/sea_ice_emissivity/footprint/shp",
                f"footprint_{flight_id}_{date}.shp",
            ),
        )
