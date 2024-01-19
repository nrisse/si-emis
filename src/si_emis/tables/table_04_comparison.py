"""
Creates table 4 in the paper.
"""

import numpy as np
import xarray as xr
from sklearn import metrics
from sklearn.feature_selection import r_regression

from si_emis.analysis.emissivity_comparison import COMPARE
from si_emis.data_preparation.airsat import airborne_filter
from si_emis.readers.airsat import read_airsat
from si_emis.readers.emissivity import read_emissivity_aircraft

# mirac-satellite combinations for comparison per flight id
FLIGHT_ID_COMBI = [
    ("183_160_mhs", "AFLUX_P5_RF08"),
    ("183_160_mhs", "AFLUX_P5_RF14"),
    ("183_160_mhs", "AFLUX_P5_RF15"),
    ("183_160_atms", "AFLUX_P5_RF08"),
    ("183_160_atms", "AFLUX_P5_RF14"),
    ("183_160_atms", "AFLUX_P5_RF15"),
]


def main():
    """
    Creates table 4 in the paper.

    Note that 89 GHz is excluded in paper, making some rows and columns
    redundant.
    """

    text = ""
    for name, flight_id in FLIGHT_ID_COMBI:
        # read satellite footprints with aircraft resampled to satellite
        ds_es = read_airsat(flight_id=flight_id)

        # filter satellite dataset for specific comparison
        ds_es = filter(ds_es, name=name)

        # get surface sensitivity
        print(ds_es.dtb.sel(surf_refl="L").mean("ix_sat_int"))

        # compute metrics for all defined combinations of MiRAC and satellites
        metric_dct = airsat_metrics(
            ds_es=ds_es,
            surf_refl="L",
            ac_channel=COMPARE[name][0]["ac"],
        )

        # calculate mean std of mirac emissivity for the MHS pixels
        subfootprint_variability = (
            ds_es.ea_e_std.sel(
                surf_refl="L", ac_channel=COMPARE[name][0]["ac"]
            )
            .mean("ix_sat_int")
            .item()
        )

        # normalized bias in percent as integer
        nbias = round(metric_dct["nbias"], 0)
        if ~np.isnan(nbias):
            nbias = int(nbias)

        # normalized rmse in percent as integer
        nrmse = round(metric_dct["nrmse"], 0)
        if ~np.isnan(nrmse):
            nrmse = int(nrmse)

        # create text for table
        text += (
            f"{flight_id.replace('_P5_', ' ')},"
            f"{metric_dct['count']},"
            f"{round(metric_dct['median_x'], 2)},"
            f"{round(metric_dct['median_y'], 2)},"
            f"{round(metric_dct['iqr_x'], 2)},"
            f"{round(metric_dct['iqr_y'], 2)},"
            # f"{round(subfootprint_variability, 2)},"
            f"{nbias},"
            f"{nrmse},"
            f"{round(metric_dct['rho'], 2)}"
            "\n"
        )

    header = ",".join(
        [
            "Campaign/RF",
            "Count",
            "Median MiRAC",
            "Median MHS",
            "IQR MiRAC",
            "IQR MHS",
            "Relative bias [%]",
            "Relative RMSE [%]",
            "Pearson correlation coefficient",
        ]
    )

    text = header + "\n" + text

    print(text)


def filter(ds_es, name):
    """
    Filter satellite dataset for specific comparison.

    Parameters
    ----------
    ds_es: xarray dataset
        Dataset with satellite footprints resampled to aircraft
    name: str
        Name of the comparison

    Returns
    -------
    ds: xarray dataset
        Filtered dataset
    """

    ixs = []
    for setup in COMPARE[name]:
        ixs.append(
            (ds_es.instrument == setup["ins"])
            & (ds_es.satellite == setup["sat"])
            & (ds_es.channel == setup["sc"])
            & (ds_es.incidence_angle >= setup["a0"])
            & (ds_es.incidence_angle < setup["a1"])
        )
    ix = xr.concat(ixs, dim="x").any("x").compute()

    ds_es = ds_es.sel(ix_sat_int=ix)

    return ds_es


def airsat_metrics(ds_es, surf_refl, ac_channel):
    """
    Calculate error metrics betweem aircraft and satellite emissivities for
    all research flights and a pre-defined combination of airborne channels
    and satellite instruments.

    The evaluated pixels correspond to the emissivity maps.

    Parameters
    ----------
    ds_es: xarray dataset
        Dataset with satellite footprints resampled to aircraft
    surf_refl: str
        Surface reflectance to use for filtering
    """

    # x: aircraft, y: satellite
    x = (
        ds_es["ea_e_mean"]
        .sel(surf_refl=surf_refl, ac_channel=ac_channel)
        .values
    )
    y = ds_es.e.sel(surf_refl=surf_refl).values

    ix_num = ~(np.isnan(x) | np.isnan(y))
    x = x[ix_num]
    y = y[ix_num]

    assert len(x) >= 10, "Not enough samples for statistics"

    metrics_dct = compute_metrics(x, y)

    return metrics_dct


def compute_metrics(x, y):
    """
    This function computes a number of metrics and returns their values in a
    dict.

    Returns
    -------
    metrics: dictionary of metrics
    """

    metrics_dct = {
        "count": len(x),
        "mean_x": np.mean(x),
        "mean_y": np.mean(y),
        "median_x": np.median(x),
        "median_y": np.median(y),
        "std_x": np.std(x),
        "std_y": np.std(y),
        "iqr_x": np.percentile(x, 75) - np.percentile(x, 25),
        "iqr_y": np.percentile(y, 75) - np.percentile(y, 25),
        "bias": np.mean(y - x),
        "mae": metrics.mean_absolute_error(x, y),
        "rmse": np.sqrt(metrics.mean_squared_error(x, y)),
        "r2": metrics.r2_score(x, y),
        "rho": r_regression(x[:, np.newaxis], y)[0],
        "nrmse": np.sqrt(metrics.mean_squared_error(x, y)) / np.mean(x) * 100,
        "nbias": np.mean((y - x) / x) * 100,
    }

    return metrics_dct


if __name__ == "__main__":
    main()
