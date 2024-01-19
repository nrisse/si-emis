"""
Print channels and polarization for satellites table
"""

from lizard.readers.band_pass import POL_MEANING, read_band_pass


def main():
    study_channels = {
        "MHS": [1, 2, 3, 4, 5],
        "ATMS": [16, 17, 18, 19, 20, 21, 22],
        "SSMIS": [17, 18, 8, 9, 10, 11],
        "AMSR2": [13, 14],
    }

    # just to read band pass
    satellites = {
        "MHS": "Metop-A",
        "ATMS": "NOAA-20",
        "SSMIS": "DMSP-F17",
        "AMSR2": "GCOM-W",
    }

    for instrument, channels in study_channels.items():
        ds = read_band_pass(instrument, satellites[instrument])
        for c in channels:
            f = ds.label.sel(channel=c).item().split(" GHz")[0]
            p = POL_MEANING[ds.polarization.sel(channel=c).item()]

            if ds.center_freq.sel(channel=c).item() < 180:
                f = f.split("$")[0]

            print(f"{instrument}\t{c}\t{f}\t{p}")


if __name__ == "__main__":
    main()
