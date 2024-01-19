"""
Read radiometer measurements from field campaign.
"""

import os

import ac3airborne
from dotenv import load_dotenv

load_dotenv()

META = ac3airborne.get_flight_segments()
CAT = ac3airborne.get_intake_catalog()
INTAKE_CACHE = dict(
    storage_options={
        "simplecache": dict(
            cache_storage=os.environ["INTAKE_CACHE"],
            same_names=True,
        )
    }
)
CRED = dict(user=os.environ["AC3_USER"], password=os.environ["AC3_PASSWORD"])


def read_from_intake(flight_id):
    """
    Reads HATPRO or MiRAC-P from ac3airborne intake catalog

    Parameters
    ----------
    flight_id

    Returns
    -------
    xarray.Dataset
    """

    # instrument name
    instrument = {
        "ACLOUD": "MiRAC-P",
        "AFLUX": "MiRAC-P",
        "MOSAiC-ACA": "HATPRO",
        "HALO-AC3": "HATPRO",
    }

    mission, platform, name = flight_id.split("_")

    if mission == "HALO-AC3":
        ds = CAT[mission][platform][instrument[mission]][flight_id](
            **INTAKE_CACHE, **CRED
        ).read()
    else:
        ds = CAT[mission][platform][instrument[mission]][flight_id](
            **INTAKE_CACHE
        ).read()

    ds["channel"] = ds["channel"] + 1

    return ds
