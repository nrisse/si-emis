"""
This script counts the number of airborne footprints within satellite 
footprints
"""

import numpy as np

from si_emis.readers.airsat import read_airsat


def main():
    """
    Counts number of airborne footprints within satellite footprint.

    Note that 89 GHz and the higher frequencies have different number of counts
    due to the different footprint locations. The count filter was applied
    by pairing the 89 GHz airborne channel with the 89-91 GHz satellite channel
    and the higher frequencies with the > 100 GHz satellite channels.
    """

    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    median_cnt_89 = []
    median_cnt_100 = []

    for flight_id in flight_ids:
        print(flight_id)
        ds = read_airsat(flight_id)

        # pre-selection
        ds = ds.sel(ix_sat_int=ds.instrument == "AMSR2")

        kwargs_below_100ghz = dict(
            surf_refl="L", ix_sat_int=ds["center_freq"] < 100, ac_channel=1
        )
        kwargs_above_100ghz = dict(
            surf_refl="L", ix_sat_int=ds["center_freq"] > 100, ac_channel=8
        )

        da_count_89 = ds["ea_e_count"].sel(**kwargs_below_100ghz)
        da_count_100 = ds["ea_e_count"].sel(**kwargs_above_100ghz)

        if len(da_count_89.ix_sat_int > 0):
            median = da_count_89.where(da_count_89 > 0).median().item()
            minimum = da_count_89.where(da_count_89 > 0).min().item()
            maximum = da_count_89.where(da_count_89 > 0).max().item()
            q10 = da_count_89.where(da_count_89 > 0).quantile(0.1).item()
            q90 = da_count_89.where(da_count_89 > 0).quantile(0.9).item()
            median_cnt_89.append(median)
            print("<100 GHz: ", median, minimum, maximum, q10, q90)

        if len(da_count_100.ix_sat_int > 0):
            median = da_count_100.where(da_count_100 > 0).median().item()
            minimum = da_count_100.where(da_count_100 > 0).min().item()
            maximum = da_count_100.where(da_count_100 > 0).max().item()
            q10 = da_count_100.where(da_count_100 > 0).quantile(0.1).item()
            q90 = da_count_100.where(da_count_100 > 0).quantile(0.9).item()
            median_cnt_100.append(median)
            print(">100 GHz: ", median, minimum, maximum, q10, q90)

    print(median_cnt_89)
    print(median_cnt_100)

    print(np.mean(median_cnt_89))
    print(np.mean(median_cnt_100))


if __name__ == "__main__":
    main()
