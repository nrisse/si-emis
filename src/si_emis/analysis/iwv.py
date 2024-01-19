"""
Prints IWV for each flight
"""

from pyPamtra.meteoSI import rh_to_iwv
import numpy as np

from si_emis.readers.profile import read_profile


def main():
    """
    Calculates IWV for radiosonde profile
    """

    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    for flight_id in flight_ids:

        # Read radiosonde profile
        profile = read_profile(flight_id)

        # Calculate IWV
        iwv = rh_to_iwv(
            relhum_lev=profile.relhum.values[np.newaxis, :]/100, 
            temp_lev=profile.temp.values[np.newaxis, :] + 273.15, 
            press_lev=profile.press.values[np.newaxis, :]*100, 
            hgt_lev=profile.hgt.values[np.newaxis, :])

        # Print IWV
        print(f"{flight_id}: {iwv:.2f} kg/m^2")


if __name__ == "__main__":
    main()
