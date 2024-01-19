"""
Prints tables for the paper.
"""

import numpy as np

from si_emis.figures.figure_04_hist import airborne_emissivity_merged


def main():
    """
    Creates tables
    """

    # read emissivity dataset for list of flights
    ds, dct_fi = airborne_emissivity_merged(
        flight_ids=[
            "ACLOUD_P5_RF23",
            "ACLOUD_P5_RF25",
            "AFLUX_P5_RF08",
            "AFLUX_P5_RF14",
            "AFLUX_P5_RF15",
        ],
        filter_kwargs=dict(drop_times=True, dtb_keep_tb=True),
    )

    ac = ["ACLOUD_P5_RF23", "ACLOUD_P5_RF25"]
    af = ["AFLUX_P5_RF08", "AFLUX_P5_RF14", "AFLUX_P5_RF15"]

    # print statistics for ACLOUD and AFLUX campaigns
    emissivity_table(ds, ac, af)


def emissivity_table(ds, ac, af):
    """
    Provides table with emissivities and uncertainties for every flight and
    every channel.

    The table is structured as follows:
        - 7 rows (5 flights and 2 campaigns)
        - 4 columns (each of the four channels)
        - cell entry: mean(mean_lower-mean_upper)+/-std

    no values occur for:
        - ACLOUD RF25, 89 and 340 GHz
        - ACLOUD RF23, 340 GHz
        - ACLOUD 340 GHz
    """

    flight_combinations = ac + af
    flight_combinations.append(ac)
    flight_combinations.append(af)

    rowname = [
        "ACLOUD RF23",
        "ACLOUD RF25",
        "AFLUX RF08",
        "AFLUX RF14",
        "AFLUX RF15",
        "ACLOUD",
        "AFLUX",
    ]

    text = ""
    for f in flight_combinations:
        for channel in [1, 7, 8, 9]:
            text += (
                uncertainty_text(
                    ds, surf_refl="L", channel=channel, flight_ids=f
                )
                + ","
            )

        # add rowname at beginning of text with a comma
        text += "\n"

    # process text
    text = text.replace(r"nan--nan", "")
    text = text.replace(r"$\pm$nan", "")
    text = text.replace(r"nan", "")

    # add rowname
    text = text.split("\n")[:-1]
    text = [f"{rowname[i]}, {text[i]}" for i in range(len(text))]
    text = "\n".join(text)

    print(text)


def uncertainty_text(ds, surf_refl, channel, flight_ids):
    """
    Prints mean, std, and mean uncertainty of emissivity at a specific
    frequency and for a specific research flight or campaign.
    """

    mean = (
        ds.e.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .mean("time")
        .item()
    )
    std = (
        ds.e.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .std("time")
        .item()
    )

    mean_unc = (
        ds.e_rel_unc.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .mean("time")
        .item()
    )

    dtb = (
        ds.dtb.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .mean("time")
        .item()
    )

    count = (
        ds.e.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .count("time")
        .item()
    )

    median = (
        ds.e.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .median("time")
        .item()
    )

    iqr = (
        ds.e.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .quantile(0.75, dim="time")
        .item()
        - ds.e.sel(
            time=np.isin(ds.flight_id, flight_ids),
            surf_refl=surf_refl,
            channel=channel,
        )
        .quantile(0.25, dim="time")
        .item()
    )

    unc = round(mean_unc, 0)
    if ~np.isnan(unc):
        unc = int(unc)

    text = (
        f"{count},"
        # f"{round(mean, 2)},"
        # f"{round(std, 2)},"
        f"{round(median, 2)},"
        f"{round(iqr, 2)},"
        # f"{round(mean_l, 2)}--{round(mean_u, 2)},"
        # f"{round(dtb, 3)}"
        f"{unc}"
    )

    return text


if __name__ == "__main__":
    main()
