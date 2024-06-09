"""
Quantify the TB noise for calm conditions during an ACLOUD flight
"""

import numpy as np
import pandas as pd
import xarray as xr

from si_emis.readers.airborne_tb import read_from_intake


def main():
    """
    TB noise quantification from clear-sky flight over open ocean during 
    ACLOUD RF10, 2 min interval.
    """

    flight_id = "ACLOUD_P5_RF10"

    ds = read_from_intake(flight_id)

    time_range = pd.date_range(
        start="2017-05-31 18:12", end="2017-05-31 18:22", freq="30s"
    )

    da_tb_std_lst = []
    for i in range(len(time_range) - 1):
        t0 = time_range[i]
        t1 = time_range[i + 1]

        da_tb_std = ds.tb.sel(time=slice(t0, t1)).std("time")

        da_tb_std_lst.append(da_tb_std)

    da_tb_std = xr.concat(da_tb_std_lst, dim="time")

    da_tb_std_mean = da_tb_std.mean("time")
    da_tb_std_std = da_tb_std.std("time")
    da_tb_std_max = da_tb_std.quantile(0.95, dim="time")

    for channel in da_tb_std_mean.channel:
        print(f"Channel {channel.values}: {np.round(da_tb_std_mean.sel(channel=channel).values, 2)} +/- {np.round(da_tb_std_std.sel(channel=channel).values, 2)} K")


if __name__ == "__main__":
    main()
