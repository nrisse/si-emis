"""
Compares surface temperature from aircraft KT-19 and satellite.
"""

from si_emis.readers.airsat import read_airsat


def main():
    """
    Surface temperature bias at satellite footprint scale. Note that this is 
    not a direct comparison of both products as footprints are not distributed
    equally in space.
    """

    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    for flight_id in flight_ids:

        ds_sat = read_airsat(flight_id)

        aircraft_ts = ds_sat.ea_ts_mean.mean().item()
        satellite_ts = ds_sat.ts.mean().item()

        delta_ts = (ds_sat.ts - ds_sat.ea_ts_mean).mean().item()

        print(f"{flight_id}: {delta_ts:.2f} K ({satellite_ts:.2f} K vs. {aircraft_ts:.2f} K)")

        # mean surface temperature from KT-19 in degrees C
        print(f"{flight_id}: {ds_sat.ea_ts_mean.mean().item() - 273.15:.2f} C")

if __name__ == '__main__':
    main()
