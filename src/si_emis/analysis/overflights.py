"""
Overflight statistics
"""

import numpy as np

from si_emis.figures.figure_08_airsat import (
    read_all_data,
    query_satellite,
)


def main():
    """ "
    Prints number of overflights etc
    """

    flights = {
        "ACLOUD": ["ACLOUD_P5_RF23", "ACLOUD_P5_RF25"],
        "AFLUX": ["AFLUX_P5_RF08", "AFLUX_P5_RF14", "AFLUX_P5_RF15"],
    }

    for mission, flight_ids in flights.items():
        print(mission)

        # read data from all flights
        ds_ea_merge, ds_es_merge = read_all_data(flight_ids)

        # print overflight statistics (how many unique granules)
        print("Number of overflights")
        print(
            "MHS",
            len(
                np.unique(
                    query_satellite(ds_es_merge, "MHS", 1, 0, 30).granule
                )
            ),
        )
        print(
            "ATMS",
            len(
                np.unique(
                    query_satellite(ds_es_merge, "ATMS", 16, 0, 30).granule
                )
            ),
        )
        print(
            "SSMIS",
            len(
                np.unique(
                    query_satellite(ds_es_merge, "SSMIS", 17, 0, 90).granule
                )
            ),
        )
        print(
            "AMSR2",
            len(
                np.unique(
                    query_satellite(ds_es_merge, "AMSR2", 13, 0, 90).granule
                )
            ),
        )

        # print number of footprints
        print("Number of footprints at 89 GHz")
        print(
            "MHS", len(query_satellite(ds_es_merge, "MHS", 1, 0, 30).granule)
        )
        print(
            "ATMS",
            len(query_satellite(ds_es_merge, "ATMS", 16, 0, 30).granule),
        )
        print(
            "SSMIS",
            len(query_satellite(ds_es_merge, "SSMIS", 17, 0, 90).granule),
        )
        print(
            "AMSR2",
            len(query_satellite(ds_es_merge, "AMSR2", 13, 0, 90).granule),
        )

        print("Number of footprints at 150-165 GHz")
        print(
            "MHS", len(query_satellite(ds_es_merge, "MHS", 2, 0, 30).granule)
        )
        print(
            "ATMS",
            len(query_satellite(ds_es_merge, "ATMS", 17, 0, 30).granule),
        )
        print(
            "SSMIS",
            len(query_satellite(ds_es_merge, "SSMIS", 9, 0, 90).granule),
        )


def print_depolarization(ds_es_merge, instrument, surf_refl):
    """
    Calculate depolarization for SSMIS and AMSR2 at 89 GHz. The resulting
    statistics is printed and a numpy array of PD is returned.

    Possible extension: return PD as a data array using either v or h-pol
    datasets.

    PD = ev - eh

    Parameters
    ----------
    ds: dataset with ix_sat_int as coordinate. It should contain emissivities,
      instrument, satellite, incidence_angle, granule, and footprint_id
      information.
    instrument: instrument name. This determines the channels that are compared

    Returns
    ----------
    depol: polarization difference numpy array.
    """

    if instrument == "SSMIS":
        chv, chh = [17, 18]
    elif instrument == "AMSR2":
        chv, chh = [13, 14]

    v = query_satellite(ds_es_merge, instrument, chv, 0, 90)
    h = query_satellite(ds_es_merge, instrument, chh, 0, 90)

    # make sure that the footprint and granule ids match between both
    assert (v.footprint_id.values == h.footprint_id.values).all()
    assert (v.granule.values == h.granule.values).all()

    depol = (
        v.e.sel(surf_refl=surf_refl).values
        - h.e.sel(surf_refl=surf_refl).values
    )

    pd_mean = depol.mean()
    pd_median = np.median(depol)
    pd_min = depol.min()
    pd_max = depol.max()

    print(
        f"Mean: {pd_mean}\nMedian: {pd_median}\nMin: {pd_min}\n"
        f"Max: {pd_max}\n"
    )

    return depol


if __name__ == "__main__":
    main()
