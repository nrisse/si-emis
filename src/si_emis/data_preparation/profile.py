"""
Script to generate merged atmospheric profile from radiosondes, noseboom, and
dropsondes. Currently, one profile is created for every research flight. The
way, the different sources are combined is defined in the profile.yaml script.

If flight segments are available, the script extracts noseboom data from
ascends and descends automatically, the user only needs to chose the
sounding by segment_id (multiple segments can be chosen)
"""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from lizard.ac3airlib import (
    get_all_flights,
    get_dropsonde_flight_ids,
    profiles,
)
from lizard.readers.dropsondes import read_dropsonde
from lizard.readers.noseboom import read_noseboom
from lizard.readers.radiosonde import read_radiosonde
from lizard.writers.figure_to_file import write_figure
from matplotlib import cm

from si_emis.readers.profile import read_profile
from si_emis.style import *


def plot_profiles(flight_ids):
    """
    Creates plot of all noseboom, dropsonde, and radiosonde profiles of one
    research flight

    Parameters
    ----------
    flight_ids: list of flight ids
    """

    flight_ids_ds = get_dropsonde_flight_ids()

    for flight_id in flight_ids:
        print(flight_id)

        # read dropsondes
        if flight_id in flight_ids_ds:
            dict_ds_dsd = read_dropsonde(flight_id)
        else:
            dict_ds_dsd = None
        ds_rsd = read_radiosonde(flight_id=flight_id)
        ds_nbm = read_noseboom(flight_id)
        if ds_nbm:
            dct_ds_nbm, segments = noseboom_profiles(ds_nbm, flight_id)
        else:
            dct_ds_nbm, segments = None, None

        # make variable names of dropsonde equal
        if flight_id in flight_ids_ds:
            if "HALO-AC3" not in flight_id:
                for sonde_id, ds in dict_ds_dsd.items():
                    dict_ds_dsd[sonde_id] = ds.rename(
                        {
                            "GPS_Alt": "gpsalt",
                            "Baro_Alt": "alt",
                            "Time": "time",
                            "Lat": "lat",
                            "Lon": "lon",
                            "Pressure": "pres",
                            "Temp": "tdry_raw",
                            "Temp_recon": "tdry",
                            "RHum": "rh_raw",
                            "RHum_recon": "rh",
                            "Wind_vel": "wspd",
                            "Wind_dir": "wdir",
                        }
                    )

        cmap1 = cm.get_cmap("tab20")
        cmap2 = cm.get_cmap("tab10")
        color_lst = [cmap1(i) for i in range(cmap1.N)]
        color_lst += [cmap2(i) for i in range(cmap2.N)]
        color_lst += [cmap1(i) for i in range(cmap1.N)]

        fig, (ax_t, ax_rh) = plt.subplots(
            1, 2, figsize=(7, 6), sharey="row", constrained_layout=True
        )

        kwargs = dict(linewidth=0, s=4)

        # plot all noseboom measurements
        if ds_nbm:
            ax_t.scatter(
                ds_nbm.T,
                ds_nbm.p,
                color="gray",
                label="entire flight",
                **kwargs,
            )
            ax_rh.scatter(
                ds_nbm.rh,
                ds_nbm.p,
                color="gray",
                label="entire flight",
                **kwargs,
            )

        # plot noseboom profiles
        i = 0  # for colors
        if ds_nbm:
            for segment in segments:
                ax_t.scatter(
                    dct_ds_nbm[segment["segment_id"]].T,
                    dct_ds_nbm[segment["segment_id"]].p,
                    color=color_lst[i],
                    label=segment["name"],
                    **kwargs,
                )
                ax_rh.scatter(
                    dct_ds_nbm[segment["segment_id"]].rh,
                    dct_ds_nbm[segment["segment_id"]].p,
                    color=color_lst[i],
                    label=segment["name"],
                    **kwargs,
                )
                i += 1

        # plot radiosonde from Ny-Alesund
        ax_t.scatter(
            ds_rsd.temp,
            ds_rsd.press,
            color="k",
            label="radiosonde NYA",
            **kwargs,
        )
        ax_rh.scatter(
            ds_rsd.rh,
            ds_rsd.press,
            color="k",
            label="radiosonde NYA",
            **kwargs,
        )

        # plot dropsondes from aircraft
        if flight_id in flight_ids_ds:
            for ds_name, ds_dsd in dict_ds_dsd.items():
                ax_t.scatter(
                    ds_dsd.tdry,
                    ds_dsd.pres,
                    color=color_lst[i],
                    label=ds_name,
                    **kwargs,
                )
                ax_rh.scatter(
                    ds_dsd.rh,
                    ds_dsd.pres,
                    color=color_lst[i],
                    label=ds_name,
                    **kwargs,
                )
                i += 1

        for ax in fig.axes:
            ax.set_yticks(np.arange(1000, 650, -10), minor=True)
        ax_t.set_ylim([1013.25, 500])
        ax_t.set_xlim([-30, 10])
        ax_rh.set_xlim([0, 100])

        ax_t.set_xlabel("Temperature [°C]")
        ax_t.set_ylabel("Pressure [hPa]")
        ax_rh.set_xlabel("Relative humidity [%]")

        lgnd = ax_rh.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            fontsize=8,
        )

        for path in lgnd.legendHandles:
            path._sizes = [50]

        ax_t.set_ylim([1013.25, 650])

        write_figure(
            fig,
            os.path.join(
                os.environ["PATH_PLT"],
                "atmospheric_profile",
                f"profile_{flight_id}",
            ),
        )

        plt.close()


def noseboom_profiles(ds, flight_id):
    """
    Extracts noseboom along vertical flights to individual profiles.

    Returns
    -------
    dct_ds_nbm: dictionairy with segment_id as key and the measured time series
    of noseboom data as values.
    """

    segments = profiles(flight_id)
    nbm_vars = ["p", "rh", "T", "h"]
    dct_ds_nbm = {}
    for segment in segments:
        dct_ds_nbm[segment["segment_id"]] = ds[nbm_vars].sel(
            time=slice(segment["start"], segment["end"])
        )

    return dct_ds_nbm, segments


def merged_profile(flight_ids):
    """
    Creates merged atmospheric profile from dropsondes, noseboom and
    radiosondes as specified in the profile.yaml file for each research
    flight. Missing values are extrapolated

    Gaps will be first backward filled (to extrapolate to surface)
    and then forward filled (to extrapolate to higher levels)

    Parameters
    ----------
    flight_ids: list of flight ids
    """

    for flight_id in flight_ids:
        print(flight_id)

        # read setting of atmospheric profile
        with open("./src/si_emis/data_preparation/profile.yaml", "r") as f:
            atm_setting = yaml.safe_load(f)[flight_id]
        atm_setting.pop("comment")

        if not any(
            [atm_setting["dsd"], atm_setting["nbm"], atm_setting["rsd"]]
        ):
            continue

        # height from 0.5 to 33,000 meters
        n_lev = 200
        z_even = np.linspace(10, 33000, n_lev, endpoint=True)
        factor = np.linspace(0.05, 1, n_lev, endpoint=True)
        z_lev = z_even * factor

        # variable names of the instruments and sondes
        var_names = {
            "atm": ["temp", "relhum", "press", "hgt"],
            # varnames used in the end
            "nbm": ["T", "rh", "p", "h"],
            "rsd": ["temp", "rh", "press", "alt"],
            "dsd": ["tdry", "rh", "pres", "gpsalt"],
        }

        # create atmopsheric dataset that gets filled with the various sources
        ds_atm_raw = xr.Dataset()
        ds_atm_raw.coords[var_names["atm"][3]] = []  # height coordinate
        ds_atm_raw[var_names["atm"][0]] = ((var_names["atm"][3]), [])
        ds_atm_raw[var_names["atm"][1]] = ((var_names["atm"][3]), [])
        ds_atm_raw[var_names["atm"][2]] = ((var_names["atm"][3]), [])

        # loop over nbm, dsd and rsd and slice data into dataset
        for source, settings in atm_setting.items():
            if settings is None:
                continue
            else:
                for setting in settings:
                    print(setting)

                    source_id, p0, p1 = setting

                    if source == "nbm":
                        ds_nbm = read_noseboom(flight_id)
                        dct_ds_nbm, segments = noseboom_profiles(
                            ds_nbm, flight_id
                        )
                        ds = dct_ds_nbm[source_id].copy(deep=True)
                    elif source == "rsd":
                        if isinstance(source_id, datetime.datetime):
                            ds_rsd = read_radiosonde(time=source_id)

                        else:
                            ds_rsd = read_radiosonde(flight_id=flight_id)
                        ds = ds_rsd.copy(deep=True)
                    else:
                        # dropsondes might be taken from a different aircraft
                        flight_id_sonde = source_id[:-3]
                        dict_ds_dsd = read_dropsonde(flight_id_sonde)

                        # make variable names of dropsonde equal
                        if "HALO-AC3" not in flight_id:
                            for sonde_id, ds in dict_ds_dsd.items():
                                dict_ds_dsd[sonde_id] = ds.rename(
                                    {
                                        "GPS_Alt": "gpsalt",
                                        "Baro_Alt": "alt",
                                        "Time": "time",
                                        "Lat": "lat",
                                        "Lon": "lon",
                                        "Pressure": "pres",
                                        "Temp": "tdry_raw",
                                        "Temp_recon": "tdry",
                                        "RHum": "rh_raw",
                                        "RHum_recon": "rh",
                                        "Wind_vel": "wspd",
                                        "Wind_dir": "wdir",
                                    }
                                )
                        ds = dict_ds_dsd[source_id].copy(deep=True)

                    # index data by pressure boundaries
                    ix = (ds[var_names[source][2]] > p1) & (
                        ds[var_names[source][2]] < p0
                    )
                    ds_sel = ds.sel(time=ix)
                    ds_sel = ds_sel.rename(
                        {
                            var_names[source][0]: var_names["atm"][0],
                            var_names[source][1]: var_names["atm"][1],
                            var_names[source][2]: var_names["atm"][2],
                            var_names[source][3]: var_names["atm"][3],
                        }
                    )

                    # make height as index and remove all other variables
                    ds_atm_raw_cur = xr.Dataset()
                    ds_atm_raw_cur.coords[var_names["atm"][3]] = ds_sel[
                        var_names["atm"][3]
                    ].values
                    ds_atm_raw_cur[var_names["atm"][0]] = (
                        (var_names["atm"][3]),
                        ds_sel[var_names["atm"][0]].values,
                    )
                    ds_atm_raw_cur[var_names["atm"][1]] = (
                        (var_names["atm"][3]),
                        ds_sel[var_names["atm"][1]].values,
                    )
                    ds_atm_raw_cur[var_names["atm"][2]] = (
                        (var_names["atm"][3]),
                        ds_sel[var_names["atm"][2]].values,
                    )

                    # append data to raw dataset
                    ds_atm_raw = xr.concat(
                        [ds_atm_raw, ds_atm_raw_cur], dim=var_names["atm"][3]
                    )

        # remove duplicate indices and nans
        _, index = np.unique(
            ds_atm_raw[var_names["atm"][3]], return_index=True
        )
        ds_atm_raw = ds_atm_raw.isel({var_names["atm"][3]: index})
        ds_atm_raw = ds_atm_raw.dropna(dim=var_names["atm"][3])
        ds_atm_raw = ds_atm_raw.isel(hgt=~np.isnan(ds_atm_raw.hgt))

        # sort index
        ds_atm_raw = ds_atm_raw.sortby(var_names["atm"][3])

        # interpolate on common grid
        ds_atm = ds_atm_raw.interp(
            coords={var_names["atm"][3]: z_lev}, method="linear"
        )

        # print number of nan's found in the different variables
        for var in var_names["atm"]:
            print(
                "Number of nan values in {}: {}".format(
                    var, np.sum(np.isnan(ds_atm[var].values)).item()
                )
            )

        # print lowest height with temperature and the temperature value
        ix_missing = ds_atm[var_names["atm"][0]].isnull()
        missing_heights = ds_atm.hgt.sel(hgt=ix_missing).values
        print(missing_heights)
        if len(missing_heights) > 0:
            print("Highest missing height: {} m".format(missing_heights[-1]))
        print(
            "Temperature of lowest available height: {} K".format(
                ds_atm[var_names["atm"][0]].sel(hgt=~ix_missing).values[0]
            ))

        # extrapolate
        # for pressure fit logarithmic curve
        if np.any(np.isnan(ds_atm[var_names["atm"][2]])):
            p_avail = ~np.isnan(ds_atm[var_names["atm"][2]])
            c = np.polyfit(
                ds_atm[var_names["atm"][3]].isel(
                    {var_names["atm"][3]: p_avail}
                ),
                ds_atm[var_names["atm"][2]].isel(
                    {var_names["atm"][3]: p_avail}
                ),
                4,
            )

            ds_atm[var_names["atm"][2]][~p_avail] = (
                c[0] * ds_atm[var_names["atm"][3]][~p_avail] ** 4
                + c[1] * ds_atm[var_names["atm"][3]][~p_avail] ** 3
                + c[2] * ds_atm[var_names["atm"][3]][~p_avail] ** 2
                + c[3] * ds_atm[var_names["atm"][3]][~p_avail] ** 1
                + c[4]
            )

        # others extrapolate linearly
        ds_atm = ds_atm.bfill(dim=var_names["atm"][3])
        ds_atm = ds_atm.ffill(dim=var_names["atm"][3])

        # save profile
        ds_atm.to_netcdf(
            os.path.join(
                os.environ["PATH_SEC"],
                "data/sea_ice_emissivity/atmospheric_profile/",
                f"profile_{flight_id}.nc",
            )
        )


def view_merged_profile(flight_ids):
    """
    Plot generated new profile as a function of pressure

    Parameters
    ----------
    flight_ids: list of flight ids
    """

    for flight_id in flight_ids:
        print(flight_id)

        ds = read_profile(flight_id)

        fig, [
            [axp_temp, axp_rh, axp_z],
            [axz_temp, axz_rh, axz_press],
        ] = plt.subplots(2, 3, figsize=(5, 6), constrained_layout=True)

        fig.suptitle(
            "black: from various sources, gray: interpolation on regular grid"
        )

        temp, rh, press, z = ["temp", "relhum", "press", "hgt"]
        kwargs = dict(linewidths=0, s=2, color="k")

        # non-interpolated profile
        # function of pressure
        axp_rh.scatter(ds[rh], ds[press], **kwargs)
        axp_temp.scatter(ds[temp], ds[press], **kwargs)
        axp_z.scatter(ds[z], ds[press], **kwargs)

        # function of height
        axz_rh.scatter(ds[rh], ds[z], **kwargs)
        axz_temp.scatter(ds[temp], ds[z], **kwargs)
        axz_press.scatter(ds[press], ds[z], **kwargs)

        # y limits for height coordinates
        for ax in [axp_temp, axp_rh, axp_z]:
            ax.set_ylim([1050, 700])
        axp_temp.set_ylabel("Pressure [hPa]")

        for ax in [axz_temp, axz_rh, axz_press]:
            ax.set_ylim([0, 33000])
        axz_temp.set_ylabel("Height [m]")

        # set xlabels and limits
        labels = [
            "Temperature [°C]",
            "Relative humidity [%]",
            "Height [m]",
            "Temperature [°C]",
            "Relative humidity [%]",
            "Pressure [hPa]",
        ]
        limits = [
            [-30, 10],
            [0, 100],
            [0, 3000],
            [-70, 10],
            [0, 100],
            [0, 1050],
        ]
        for i, ax in enumerate(
            [axp_temp, axp_rh, axp_z, axz_temp, axz_rh, axz_press]
        ):
            ax.set_xlabel(labels[i])
            ax.set_xlim(limits[i])

        write_figure(
            fig,
            os.path.join(
                os.environ["PATH_PLT"],
                "atmospheric_profile",
                f"profile_merged_{flight_id}",
            ),
        )

        plt.close()


if __name__ == "__main__":
    flight_ids = get_all_flights(
        ["ACLOUD", "AFLUX", "MOSAiC-ACA", "HALO-AC3"], "P5"
    )
    flight_ids.append("HALO-AC3_HALO_RF10")

    flight_ids =[
            "ACLOUD_P5_RF23",
            "ACLOUD_P5_RF25",
            "AFLUX_P5_RF08",
            "AFLUX_P5_RF14",
            "AFLUX_P5_RF15",
        ]

    # plot_profiles(flight_ids)
    merged_profile(flight_ids)
    view_merged_profile(flight_ids)
