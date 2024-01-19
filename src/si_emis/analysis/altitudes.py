"""
MiRAC flight altitudes during the emissivity flights
"""

from si_emis.readers.emissivity import read_emissivity_aircraft
from si_emis.data_preparation.airsat import airborne_filter


def main():
    """
    Overview of flight altitudes for emissivity flights
    """

    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    for flight_id in flight_ids:
        ds = read_emissivity_aircraft(flight_id)
        ds = airborne_filter(ds, drop_times=True)

        # print flight altitude statistics
        alt_mean = ds.ac_alt.mean("time").item()
        alt_min = ds.ac_alt.min("time").item()
        alt_max = ds.ac_alt.max("time").item()
        alt_median = ds.ac_alt.median("time").item()

        # frequency of height ranges
        below_1km = (ds.ac_alt < 500).mean("time").item() * 100
        above_2km = (ds.ac_alt > 2500).mean("time").item() * 100

        print(
            f"{flight_id}: {alt_mean:.0f} m, "
            f"{alt_min:.0f} m, {alt_max:.0f} m, {alt_median:.0f} m, "
            f"{below_1km}% below 1 km, {above_2km}% above 2 km"
        )


if __name__ == "__main__":
    main()
