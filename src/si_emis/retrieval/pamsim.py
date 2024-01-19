"""
PAMTRA simulation for surface emissivity calculation
"""

from __future__ import division

import argparse

import numpy as np
import pyPamtra
import xarray as xr
from lizard.bitflags import read_flag_xarray
from lizard.pamtools.pamtra_tools import PamtraSurface, pam_to_xarray
from lizard.pamtools.profile import PAM_VARS
from lizard.readers.band_pass import read_band_pass, read_band_pass_combination
from lizard.readers.brightness_temperature import read_tb
from lizard.readers.sat_at_aircraft import read_sat_at_aircraft_unique

from si_emis.readers.profile import read_profile
from si_emis.retrieval.makesetups import read_setups
from si_emis.style import sensors
from si_emis.writers.pamtra_simulation import write_pamtra_simulation

PAMTRA_PARALLEL = dict(
    pp_deltaX=13, pp_deltaY=1, pp_deltaF=1, pp_local_workers=13
)


def run_setups(file, platform, setup_names=None, all_setup_names=True):
    """
    Runs pamtra simulation for all configurations in the given setup file

    Parameters
    ----------
    file: setup file path
    platform: aircraft or satellite
    setup_names: list of setups to be executed (e.g. 00, 01, 02, ...)
    all_setup_names: whether to run all available setup names
    """

    setups = read_setups(file)

    if setup_names != [""]:
        all_setup_names = False
        print(f"Executing setups: {setup_names}")

    for name, setup in setups.items():
        if all_setup_names:
            pass
        elif name in setup_names:
            pass
        else:
            print(f"++++++\nWarning: Skipping {platform} setup {name}\n")
            continue

        print(f"++++++\nRunning {platform} setup {name}:\n{setup}\n")

        if platform == "aircraft":
            run_airsim(setup)

        elif platform == "satellite":
            run_satsim(setup)

        print("++++++\n")


def make_pam(ds, ds_surf, setup, obs_heights: list = None, axis: str = None):
    """
    Create pamtra object based on the following input:
    - single atmospheric profile with clear-sky as a function of hgt
    - other surface properties as a function of a 1D variable

    Parameters
    ----------
    ds: xarray.Dataset of atmospheric profile (press - hPa, temp - C, hgt - m,
     relhum - %)
    ds_surf: xarray.Dataset that contains surface parameters and ancillary
     parameters (time, lat, lon, ts, land, sea_ice, surf_alt,
    setup: setup dict
    obs_heights: observation height as a function of axis. Default: 833 km.
    axis: the coordinate axis of ds_surface along which the pamtra simulation
     will be conducted (e.g. time). Default: time.

    Returns
    -------
    pam: pyPamtra object with atmospheric profile and in-situ data
    """

    if axis is None:
        axis = "time"

    if obs_heights is None:
        obs_heights = [
            ds_surf.surf_alt,
            xr.full_like(ds_surf[axis], fill_value=833000),
        ]

    # convert units
    ds["press"] = ds["press"] * 100
    ds["temp"] = ds["temp"] + 273.15

    ds["time"] = ds_surf.time

    ds["lat"] = ds_surf.lat
    ds["lon"] = ds_surf.lon
    ds["ts"] = ds_surf.ts
    ds["land"] = ds_surf.land
    ds["sea_ice"] = ds_surf.sea_ice
    ds["surf_alt"] = ds_surf.surf_alt

    # outlevel
    ds.coords["outlevel"] = np.arange(0, len(obs_heights))
    ds["obs_height"] = xr.concat(obs_heights, dim=ds.outlevel)

    # rename variables
    rename_dct = {
        "time": "timestamp",
        "land": "sfc_slf",
        "sea_ice": "sfc_sif",
        "ts": "groundtemp",
    }
    ds = ds.rename(rename_dct)

    if axis in rename_dct.keys():
        axis = rename_dct[axis]

    # (axis, none, hgt) to (x, y, z)
    ds["grid_x"] = (axis, np.arange(0, len(ds[axis])))
    ds = ds.swap_dims({axis: "grid_x"}).reset_coords()

    ds.coords["grid_y"] = np.array([0])

    ds["grid_z"] = ("hgt", np.arange(0, len(ds.hgt)))
    ds = ds.swap_dims({"hgt": "grid_z"}).reset_coords()

    # expand dimensions of variables
    ds["lon"] = ds["lon"].expand_dims("grid_y")
    ds["lat"] = ds["lat"].expand_dims("grid_y")
    ds["surf_alt"] = ds["surf_alt"].expand_dims("grid_y")
    ds["obs_height"] = ds["obs_height"].expand_dims("grid_y")
    ds["timestamp"] = ds["timestamp"].expand_dims("grid_y")
    ds["sfc_slf"] = ds["sfc_slf"].expand_dims("grid_y")
    ds["sfc_sif"] = ds["sfc_sif"].expand_dims("grid_y")
    ds["groundtemp"] = ds["groundtemp"].expand_dims("grid_y")
    ds["hgt"] = ds["hgt"].expand_dims(
        {"grid_x": ds.grid_x, "grid_y": ds.grid_y}
    )
    ds["temp"] = ds["temp"].expand_dims(
        {"grid_x": ds.grid_x, "grid_y": ds.grid_y}
    )
    ds["relhum"] = ds["relhum"].expand_dims(
        {"grid_x": ds.grid_x, "grid_y": ds.grid_y}
    )
    ds["press"] = ds["press"].expand_dims(
        {"grid_x": ds.grid_x, "grid_y": ds.grid_y}
    )

    # adjust order of dimensions
    ds = ds.transpose("grid_x", "grid_y", "grid_z", "outlevel")

    # convert time format
    ds["timestamp"] = (
        ("grid_x", "grid_y"),
        (ds.timestamp.values - np.datetime64("1970-01-01"))
        .astype("timedelta64[s]")
        .astype("int"),
    )

    # define surface reflection and model based on input
    ds = PamtraSurface(ds=ds, **setup["surface"]).ds

    # add offset to atmospheric profile and surface temperature as defined
    # in the setup file
    if "offset" in setup.keys():
        for var, offset in setup["offset"].items():
            ds[var] = ds[var] + offset

    # remove unrealistic values
    ds["relhum"] = ds["relhum"].where(ds["relhum"] < 100, 100)
    ds["relhum"] = ds["relhum"].where(ds["relhum"] > 0, 0)

    # remove atmospheric layers beneath the surface
    # pamtra drops profiles that have any nan value, therefore fill
    # with neighboring layer (then layers repeat)
    atm_vars = ["press", "relhum", "temp", "hgt"]
    for v in atm_vars:
        ds[v] = ds[v].where(ds["hgt"] > ds.surf_alt)
    ds[["press", "relhum", "temp"]] = ds[["press", "relhum", "temp"]].bfill(
        "grid_z"
    )

    # this adds thin height layers below the lowest height above ground
    # pressure, humidity, and temperature of thin layers doesn't change
    ix = ds["hgt"].isnull().idxmin("grid_z")  # first layer above surface
    ds["hgt"][:, :, 0] = ds["hgt"].sel(grid_z=ix) - 0.1  # lowest layer
    ds["hgt"] = ds["hgt"].interpolate_na("grid_z")

    # add empty winds
    ds["wind10u"] = xr.zeros_like(ds["groundtemp"])
    ds["wind10v"] = xr.zeros_like(ds["groundtemp"])

    pam_profile = dict()
    for var in PAM_VARS:
        if var == "hydro_q":
            continue
        pam_profile[var] = ds[var].values

    pam = pyPamtra.pyPamtra()
    pam.df.addHydrometeor(
        (
            "ice",
            -99.0,
            -1,
            917.0,
            130.0,
            3.0,
            0.684,
            2.0,
            3,
            1,
            "mono_cosmo_ice",
            -99.0,
            -99.0,
            -99.0,
            -99.0,
            -99.0,
            -99.0,
            "mie-sphere",
            "heymsfield10_particles",
            0.0,
        )
    )
    pam.createProfile(**pam_profile)

    pam.nmlSet.update(setup["nmlSet"])
    pam.addIntegratedValues()

    return pam


def select_times(ds_tb, times):
    """
    Select only times within specified time windows including their left and
    excluding right boundary.

    The times should be given in a 2D numpy array or list of shape (N, 2)
    with the number of time windows N and the second dimension provind the
    start and end times. Date type of times is datetime64[s].

    Parameters
    ----------
    ds_tb: dataset that has a time dimension (e.g. tb data)
    times: time windows

    Returns
    -------
    ds_tb: same dataset with only the selected time windows
    """

    times = np.array(times).flatten()
    ix = np.searchsorted(ds_tb.time.values, times.astype("datetime64[s]"))
    ix = np.hstack([np.arange(i1, i2) for i1, i2 in zip(ix[::2], ix[1::2])])
    ds_tb = ds_tb.isel(time=ix)

    return ds_tb


def run_airsim(setup):
    """
    Write atmospheric profile to PAMTRA object and run forward simulation

    The simulation is run for each channel separately, because these parameters
    are channel-dependent:
      - lon, lat (needed in telsem2)
      - surface altitude (over terrain)
      - surface temperature (KT-19 is matched with slanted MiRAC-A view)
      - surface type (due to slanted view, especially in fjords)

    After the simulation is run for each channel, all simulations are merged
    again along the frequency dimension. Note that some variables may depend
    on the frequency now, such as the iwv (over land where MiRAC-A footprint
    sees a different terrain height than MiRAC-P at a given time).
    """

    mission, platform, name = setup["general"]["flight_id"].split("_")

    ds_atm = read_profile(setup["general"]["flight_id"])
    ds_tb = read_tb(setup["general"]["flight_id"])

    ds_tb = read_flag_xarray(ds_tb, "surf_type", drop=True)

    # select time windows including their left and excluding right boundary
    ds_tb = select_times(ds_tb, times=setup["general"]["times"])

    lst_ds_pam = []
    for channel in ds_tb.channel:
        ds_tb_c = ds_tb.sel(channel=channel).reset_coords(drop=True)

        # read band pass information for averaging frequencies
        sen = sensors[mission][platform]
        if sen["sensor2"] is None:
            ds_bp = read_band_pass(sen["sensor1"])  # hamp on halo
        else:
            ds_bp = read_band_pass_combination(**sen)  # polar 5
        frequencies = np.unique(ds_bp.avg_freq.sel(channel=channel).values)

        # atmosphere to shape as expected by pamtra
        pam = make_pam(
            ds=ds_atm.copy(deep=True),
            ds_surf=ds_tb_c,
            setup=setup,
            obs_heights=[ds_tb_c.surf_alt, ds_tb_c.ac_alt],
            axis="time",
        )

        pam.runParallelPamtra(frequencies, **PAMTRA_PARALLEL)
        ds_pam = pam_to_xarray(pam, split_angle=True)

        # drop variables
        ds_pam = ds_pam.drop(
            (
                "hydro_class",
                "model_i",
                "model_j",
                "nlyrs",
                "lon",
                "lat",
                "sfc_type",
                "groundtemp",
                "sfc_salinity",
                "sfc_sif",
                "sfc_slf",
                "sfc_model",
                "sfc_refl",
                "wind10u",
                "wind10v",
                "hydro_wp",
            )
        )

        # add global attributes
        ds_pam.attrs.update(setup["surface"])
        ds_pam.attrs["flight_id"] = setup["general"]["flight_id"]
        ds_pam.attrs.update(
            dict(
                offset_groundtemp=setup["offset"]["groundtemp"],
                offset_groundtemp_unit="K",
                offset_temp=setup["offset"]["temp"],
                offset_temp_unit="K",
                offset_relhum=setup["offset"]["relhum"],
                offset_relhum_unit="%",
            )
        )

        lst_ds_pam.append(ds_pam)

    # merge simulations
    ds_pam = xr.concat(
        lst_ds_pam,
        dim="frequency",
        data_vars=["tb", "iwv", "emissivity"],
        coords=["frequency"],
        compat="override",
        combine_attrs="override",
    )

    write_pamtra_simulation(ds_pam, setup["general"]["outfile"])


def run_satsim(setup):
    """
    Run PAMTRA simulation for GPM satellites with overlapping footprint along
    flight track.
    The simulation is run for each instrument separately.
    """

    ds_atm = read_profile(setup["general"]["flight_id"])
    ds_tb = read_sat_at_aircraft_unique(setup["general"]["flight_id"])

    ds_tb = ds_tb.rename(
        {"lon_sat": "lon", "lat_sat": "lat", "scan_time": "time"}
    )

    # run pamtra for each instrument
    instruments = np.unique(ds_tb["instrument"].values)
    lst_ds_pam = []
    for instrument in instruments:
        print(f"{instrument}")

        ds_tb_c = ds_tb.sel(ix_sat_int=(ds_tb.instrument == instrument))

        # get averaging frequencies for pamtra
        frequencies = np.unique(ds_tb_c.avg_freq.values)

        pam = make_pam(
            ds=ds_atm.copy(deep=True),
            ds_surf=ds_tb_c,
            setup=setup,
            obs_heights=[
                ds_tb_c.surf_alt,
                xr.full_like(ds_tb_c.surf_alt, fill_value=833000),
            ],
            axis="ix_sat_int",
        )

        pam.runParallelPamtra(frequencies, **PAMTRA_PARALLEL)
        ds_pam = pam_to_xarray(pam, split_angle=True)

        # drop variables
        ds_pam = ds_pam.drop(
            (
                "hydro_class",
                "model_i",
                "model_j",
                "nlyrs",
                "lon",
                "lat",
                "sfc_type",
                "groundtemp",
                "sfc_salinity",
                "sfc_sif",
                "sfc_slf",
                "sfc_model",
                "sfc_refl",
                "wind10u",
                "wind10v",
                "hydro_wp",
            )
        )

        # add global attributes
        ds_pam.attrs.update(setup["surface"])
        ds_pam.attrs["flight_id"] = setup["general"]["flight_id"]

        # add the observation index again to match simulation with tb lateron
        ds_pam["ix_sat_int"] = ("grid_x", ds_tb_c.ix_sat_int.values)

        lst_ds_pam.append(ds_pam)

    # merge simulations
    ds_pam = xr.concat(
        lst_ds_pam,
        dim="grid_x",
        data_vars=["tb", "iwv", "emissivity"],
        coords=["grid_x"],
        compat="override",
        combine_attrs="override",
    )

    write_pamtra_simulation(ds_pam, setup["general"]["outfile"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "flight_id",
        help="ac3airborne flight id (e.g. " "ACLOUD_P5_RF23)",
        type=str,
    )
    parser.add_argument(
        "platform",
        help="observation platform",
        type=str,
        choices=["aircraft", "satellite"],
    )
    parser.add_argument(
        "--setup_names",
        help="only this selection of setup " "names will be simulated",
        nargs="+",
        required=False,
    )
    parser.add_argument(
        "--all_setup_names",
        help="all setups that are " "available will be" "simulated",
        type=bool,
        choices=[True, False],
        required=False,
    )
    args = parser.parse_args()

    if args.platform == "satellite":
        run_setups(
            file=f"./retrieval/setups/setup_sat_{args.flight_id}.yaml",
            platform=args.platform,
            setup_names=args.setup_names,
            all_setup_names=args.all_setup_names,
        )

    elif args.platform == "aircraft":
        run_setups(
            file=f"./retrieval/setups/setup_{args.flight_id}.yaml",
            platform=args.platform,
            setup_names=args.setup_names,
            all_setup_names=args.all_setup_names,
        )
