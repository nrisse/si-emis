"""
Read file with emissivity and additionally data.
"""

import os

import ac3airborne
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

META = ac3airborne.get_flight_segments()


def read_emissivity_aircraft(flight_id, without_offsets=True):
    """
    Read emissivity dataset.

    Parameters
    ----------
    flight_id: research flight id from ac3airborne.
    with_offsets: reads also emissivity and emissivity sensitivity calculated
      under the perturbed profiles. (this was used in reflectivity.py, where
      the Tb was calculated at a different channel with a perturbed atmosphere
      and emissivity rather than the emissivity under unperturbed conditions.
      Otherwise one could not match the PAMTRA simulation under perturbed
      conditions with the emissivity...)
    """

    mission, platform, name = flight_id.split("_")
    date = META[mission][platform][flight_id]["date"].strftime("%Y%m%d")

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/emissivity",
            f"emissivity_{flight_id}_{date}.nc",
        )
    )

    if without_offsets:
        has_offset = (ds.offset != 0).any("variable")
        ds[["dtb", "e"]] = (
            ds[["dtb", "e"]]
            .sel(i_offset=~has_offset)
            .squeeze()
            .reset_coords(drop=True)
        )

    return ds


def read_emissivity_satellite(flight_id):
    """
    Read emissivity dataset from satellite footprints that align with aircraft.

    Parameters
    ----------
    flight_id: research flight id from ac3airborne.
    """

    mission, platform, name = flight_id.split("_")
    date = META[mission][platform][flight_id]["date"].strftime("%Y%m%d")

    ds = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/",
            "emissivity",
            f"emissivity_sat_{flight_id}_{date}.nc",
        )
    )

    return ds
