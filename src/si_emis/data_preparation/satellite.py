"""
This script is the first step of the comparison between satellites and P5
observations. It collects the satellite TB and viewing angle for each research
flight along the P5 flight track of the same day.

Following steps:
    - comparison of TBs (especially at 89 GHz, which is the main purpose here
      but also 183 GHz comparisons are very important)
    - calculation of satellite TB from aircraft TB by weighting the airborne
      measurements with respect to the distance to the satellite footprint
    - temporal variability during flight time
    - calculation of satellite emissivity from aircraft emissivity (weighted)
        - this would require satellite emissivities first, so no sure... these
          will not be available along track, but for specific flights. maybe
          this script can be re-used for this purpose then!

From command line:
nohup bash -c "python -m data_preparation.satellite atrack; python -m \
data_preparation.satellite unique; python -m data_preparation.satellite\
at_time" &
"""

import argparse
import os

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from lizard.ac3airlib import day_of_flight
from lizard.readers.band_pass import read_band_pass
from lizard.readers.era5 import read_era5_single_levels
from lizard.readers.footprint import read_footprint
from lizard.readers.gpm_l1c import get_granules, read_gpm_l1c
from lizard.readers.sat_at_aircraft import read_sat_at_aircraft
from scipy.spatial import KDTree

from si_emis.readers.ne23 import read_esa_cmems

FLIGHT_IDS = [
    "ACLOUD_P5_RF23",
    "ACLOUD_P5_RF25",
    #'AFLUX_P5_RF08',
    #'AFLUX_P5_RF14',
    #'AFLUX_P5_RF15',
]

# satellites/instruments used
INS_SAT = [
    ("SNPP", "ATMS"),
    ("NOAA-20", "ATMS"),
    ("NOAA-19", "MHS"),
    ("NOAA-18", "MHS"),
    ("Metop-A", "MHS"),
    ("Metop-B", "MHS"),
    ("Metop-C", "MHS"),
    ("DMSP-F18", "SSMIS"),
    ("DMSP-F17", "SSMIS"),
    ("DMSP-F16", "SSMIS"),
    ("GCOM-W", "AMSR2"),
]


def flag_gpml1c(ds, verbose=False):
    """
    Flags gpm l1c based on the provided quality mask and an additional filter
    for corrupt scans. A very strict filter is applied that only uses
    thos pixels of "good" quality (quality flag = 0). The filter for corrupt
    scans sets those scans to nan where the across-track standard deviation is
    smaller than 0.1 K. Such low values are unrealistic.

    See documentation for quality flag meanings:
    https://arthurhou.pps.eosdis.nasa.gov/Documents/L1C_ATBD_v1.9_GPMV07.pdf#
    Sensor-specific quality flags:
        p29 (SSMIS), p35 (AMSR2), p38 (ATMS), p44 (MHS)
    Errors (negative flag): tb is set to missing values (-9999)
    Warnings (positive flag): tb is retained
    """

    # set missing tb to nan (this included all errors)
    ds["tb"] = ds.tb.where(ds["tb"] > 0)

    # filter calculation: True=remove values
    fltr = ds.tb.std("y") < 0.1

    # number of pixels/scans that are filtered
    n = np.sum(fltr.values)

    # data range before filter
    tb00 = ds["tb"].min().item()
    tb11 = ds["tb"].max().item()

    # apply filters
    ds["tb"] = ds.tb.where(~fltr)

    if verbose:
        print(f"Value range of instrument: {tb00}..{tb11}")
        if n > 0:
            tb0 = ds["tb"].min().item()
            tb1 = ds["tb"].max().item()
            print(f"New value range of instrument: {tb0}..{tb1}")

            flags = np.unique(ds.quality.where(fltr).values)
            flags = flags[~np.isnan(flags)]
            print(f"{n} scans removed by own scan line filter: {flags}")

    return ds


def aircraft_track():
    """
    Collects satellite overpasses along the aircraft flight track.
    """

    proj_in = ccrs.PlateCarree()
    proj_out = ccrs.epsg("3413")

    res = 20000

    for flight_id in FLIGHT_IDS:
        print(flight_id)

        ds_fpr = read_footprint(flight_id)

        roi = [
            ds_fpr.lon.min().item() - 1,
            ds_fpr.lon.max().item() + 1,
            ds_fpr.lat.min().item() - 1,
            ds_fpr.lat.max().item() + 1,
        ]

        granules_dct = get_granules(
            ins_sat=INS_SAT, date=day_of_flight(flight_id), roi=roi
        )

        lst_da_tb = []
        lst_da_lon = []
        lst_da_lat = []
        lst_da_inc = []
        lst_scan_time = []
        lst_dist = []
        lst_da_alt = []
        lst_instr = []
        lst_sat = []
        lst_ch = []
        lst_granule = []
        lst_ds_bp = []
        lst_footprint_id = []
        for key, granules in granules_dct.items():
            for granule in granules:
                satellite, instrument = key.split("_")

                print(f"{instrument}/{satellite}: granule {granule}")

                ds = read_gpm_l1c(instrument, satellite, granule, roi=roi)

                ds = flag_gpml1c(ds)

                # get the closest observation to airborne footprint
                swath_shape = ds["lon"].shape
                swath = proj_out.transform_points(
                    proj_in,
                    ds["lon"].values.flatten(),
                    ds["lat"].values.flatten(),
                )[:, :2]

                tree = KDTree(swath)

                footprint = proj_out.transform_points(
                    proj_in,
                    ds_fpr["lon"].values.flatten(),
                    ds_fpr["lat"].values.flatten(),
                )[:, :2]

                distance, ix = tree.query(footprint, distance_upper_bound=res)
                ix[ix == np.prod(swath_shape)] = 0
                ix_x, ix_y = np.unravel_index(ix, swath_shape)

                # get observations at indices
                ix_x = xr.DataArray(
                    ix_x.reshape(ds_fpr.lon.shape),
                    dims=ds_fpr.lon.dims,
                    coords=ds_fpr.lon.coords,
                    name="x",
                )
                ix_y = xr.DataArray(
                    ix_y.reshape(ds_fpr.lon.shape),
                    dims=ds_fpr.lon.dims,
                    coords=ds_fpr.lon.coords,
                    name="y",
                )

                ds_sat = ds.isel(x=ix_x, y=ix_y).reset_coords(drop=True)

                # store distance information
                da_dist = xr.DataArray(
                    distance.reshape(ds_fpr.lon.shape),
                    dims=ds_fpr.lon.dims,
                    coords=ds_fpr.lon.coords,
                    name="dist",
                )

                # flag where distance is inf
                ds_sat = ds_sat.where(da_dist != np.inf)

                for channel in ds.channel.values:
                    lst_da_tb.append(
                        ds_sat.tb.sel(channel=channel).reset_coords(drop=True)
                    )
                    lst_da_lon.append(ds_sat.lon)
                    lst_da_lat.append(ds_sat.lat)
                    lst_da_inc.append(np.abs(ds_sat.incidence_angle))
                    lst_scan_time.append(ds_sat.scan_time)
                    lst_footprint_id.append(ds_sat.footprint_id)
                    lst_dist.append(da_dist)
                    lst_da_alt.append(ds_sat.sc_alt)

                    lst_instr.append(instrument)
                    lst_sat.append(satellite)
                    lst_ch.append(channel)
                    lst_granule.append(granule)
                    lst_ds_bp.append(
                        read_band_pass(instrument, satellite)
                        .sel(channel=channel)
                        .reset_coords(drop=True)
                    )

        # combine information into single dataset
        ix = pd.Index(np.arange(0, len(lst_da_tb)), name="i_obs")
        ds_fpr["tb_sat"] = xr.concat(lst_da_tb, dim=ix)
        ds_fpr["incidence_angle"] = xr.concat(lst_da_inc, dim=ix)
        ds_fpr["lon_sat"] = xr.concat(lst_da_lon, dim=ix)
        ds_fpr["lat_sat"] = xr.concat(lst_da_lat, dim=ix)
        ds_fpr["alt_sat"] = xr.concat(lst_da_alt, dim=ix)
        ds_fpr["distance"] = xr.concat(lst_dist, dim=ix)
        ds_fpr["scan_time"] = xr.concat(lst_scan_time, dim=ix)
        ds_fpr["footprint_id"] = xr.concat(lst_footprint_id, dim=ix)

        ds_fpr["instrument"] = ("i_obs", lst_instr)
        ds_fpr["satellite"] = ("i_obs", lst_sat)
        ds_fpr["channel"] = ("i_obs", lst_ch)
        ds_fpr["granule"] = ("i_obs", lst_granule)
        ds_fpr[list(lst_ds_bp[0])] = xr.concat(lst_ds_bp, dim=ix)

        # remove observations that are all nan
        ds_fpr = ds_fpr.sel(
            i_obs=~(ds_fpr["tb_sat"].isnull()).all(["time", "view_ang"])
        )

        # write to file
        ds_fpr.to_netcdf(
            os.path.join(
                os.environ["PATH_SEC"],
                "data/sea_ice_emissivity/sat_flight_track",
                f"{flight_id}_gpm_l1c.nc",
            )
        )


def sat_footprint_at_time():
    """
    Computes a boolean mask sc2ac as a function of (ix_sat_int, time, channel)
    that matches spacecraft measurements to aircraft measurements.
    ix_sat_int: used to give unique ids to satellite footprints for every
      aircraft research flight
    time: times at which aircraft measures
    channel: channel-dependence due to backward-viewing airborne channel 1 and
      nadir-viewing channels >=2

    The satellite index is composed of:
      - instrument
      - satellite
      - granule
      - channel
      - footprint_id (unique within granule)

    Application: this mask allows direct indexing and averaging of airborne
      tb/emissivity to satellite tb/emissivity, histograms of airborne tb for
      a set of satellite footprints, ...

    Warning: the final mask might become very large with approx.
      20,000x20,000x10= 4,000 Mio boolean values. Therefore, any computation
      involving this mask has to be computed with dask. Just the boolean mask
      has a size of about 1 GB
    """

    proj_in = ccrs.PlateCarree()
    proj_out = ccrs.epsg("3413")

    res = 20000

    for flight_id in FLIGHT_IDS:
        print(flight_id)

        ds_fpr = read_footprint(flight_id)

        roi = [
            ds_fpr.lon.min().item() - 1,
            ds_fpr.lon.max().item() + 1,
            ds_fpr.lat.min().item() - 1,
            ds_fpr.lat.max().item() + 1,
        ]

        granules_dct = get_granules(
            ins_sat=INS_SAT, date=day_of_flight(flight_id), roi=roi
        )

        lst_instr = []
        lst_sat = []
        lst_ch = []
        lst_granule = []
        lst_footprint_id = []
        lst_da_at_time = []
        lst_da_dist = []
        for key, granules in granules_dct.items():
            for granule in granules:
                satellite, instrument = key.split("_")

                print(f"{instrument}/{satellite}: granule {granule}")

                ds = read_gpm_l1c(
                    instrument,
                    satellite,
                    granule,
                    roi=roi,
                )

                ds = flag_gpml1c(ds)

                # get the closest observation to airborne footprint
                swath_shape = ds["lon"].shape
                swath = proj_out.transform_points(
                    proj_in,
                    ds["lon"].values.flatten(),
                    ds["lat"].values.flatten(),
                )[:, :2]

                tree = KDTree(swath)

                footprint = proj_out.transform_points(
                    proj_in,
                    ds_fpr["lon"].values.flatten(),
                    ds_fpr["lat"].values.flatten(),
                )[:, :2]

                distance, ix = tree.query(footprint, distance_upper_bound=res)

                # loop over all unique satellite indices
                ix_unique = list(np.unique(ix))

                # remove the index that corresponds to nan
                try:
                    ix_unique.remove(np.prod(swath_shape))
                except ValueError:
                    pass

                # loop over every footprint that matches aircraft
                for ix_un in ix_unique:
                    # boolean mask of airborne times that match the footprint
                    da_at_time = xr.zeros_like(ds_fpr.lon, dtype="bool")
                    da_at_time.attrs = {}
                    da_at_time = da_at_time.rename("at_time")
                    da_at_time[:] = (ix == ix_un).reshape(ds_fpr.lon.shape)

                    # distance to satellite footprint
                    da_dist = xr.zeros_like(ds_fpr.lon, dtype="int")
                    da_dist.attrs = {}
                    da_dist = da_dist.rename("distance")
                    da_dist[:] = np.around(
                        distance.reshape(ds_fpr.lon.shape), 0
                    ).astype("int")

                    # this gets satellite measurements at current footprint
                    ix_x, ix_y = np.unravel_index(ix_un, swath_shape)
                    ds_sat = ds.isel(x=ix_x, y=ix_y).reset_coords(drop=True)

                    # loop over every channel
                    for channel in ds_sat.channel.values:
                        # this avoids the storing of footprints without tb
                        tb = ds_sat.tb.sel(channel=channel).item()
                        if np.isnan(tb):
                            print("skip")
                            continue

                        lst_footprint_id.append(ds_sat.footprint_id.item())
                        lst_instr.append(instrument)
                        lst_sat.append(satellite)
                        lst_ch.append(channel)
                        lst_granule.append(granule)
                        lst_da_at_time.append(da_at_time)
                        lst_da_dist.append(da_dist)

        # create satellite index for this research flight
        df = pd.DataFrame(
            data=np.array(
                [lst_instr, lst_sat, lst_ch, lst_granule, lst_footprint_id]
            ).T
        )
        df = df[0] + "_" + df[1] + "_" + df[2] + "_" + df[3] + "_" + df[4]

        # boolean mask
        ds_at_time = xr.concat(
            lst_da_at_time,
            dim="ix_sat_int",
            coords=[],
            compat="override",
            join="override",
            combine_attrs="drop",
        )
        ds_at_time = ds_at_time.to_dataset()
        ds_at_time.coords["ix_sat_int"] = df.index.values

        # distance
        da_dist = xr.concat(
            lst_da_dist,
            dim="ix_sat_int",
            coords=[],
            compat="override",
            join="override",
            combine_attrs="drop",
        )
        da_dist.coords["ix_sat_int"] = df.index.values

        # assign
        ds_at_time["distance"] = da_dist

        # assign basic information
        ds_at_time["ix_sat"] = ("ix_sat_int", df.values)
        ds_at_time["footprint_id"] = ("ix_sat_int", lst_footprint_id)
        ds_at_time["instrument"] = ("ix_sat_int", lst_instr)
        ds_at_time["satellite"] = ("ix_sat_int", lst_sat)
        ds_at_time["channel"] = ("ix_sat_int", lst_ch)
        ds_at_time["granule"] = ("ix_sat_int", lst_granule)

        # write to file
        ds_at_time.to_netcdf(
            os.path.join(
                os.environ["PATH_SEC"],
                "data/sea_ice_emissivity/sat_flight_track",
                f"{flight_id}_gpm_l1_sc2ac.nc",
            )
        )


def unique_footprints(flight_id, t0=None, t1=None):
    """
    Converts satellite overflights as a function of aircraft flight time to
    unique satellite samples of (instrument, satellite, channel, granule,
    footprint_id). This can be then used as input for pamtra simulations and
    enables to simulate the same channels of different overflights or
    footprints at the same time.

    Requirements: pre-computed satellite overflights (product of overpass of a
     satellite and channel combination times the number of channels) as a
     function of aircraft time and aircraft viewing angle (0 or -25°)

    TODO: instead of giving t0 and t1, extend it to list of times (this makes
      sense if it has to be executed again for the emissivity flights. also
      it might make pamtra simulations much slower, if all footprints are
      simulated. but maybe better too many in the beginning than not enough.
      problem: I would need already times from the setup used for pamtra to
      reduce this step!

    Parameters
    ----------
    flight_id: ac3airborne flight_id.
    t0: start of aircraft time of interest
    t1: end of aircraft time of interest

    Returns
    -------
    ds_tb: dataset with unique satellite observations during airborne
     observation period including satellite tb and additional information.
    """

    # read satellite overpasses as a function of aircraft time
    ds_tb = read_sat_at_aircraft(flight_id)

    if t0 is None and t1 is None:
        t0 = ds_tb.time.values[0]
        t1 = ds_tb.time.values[-1]

    ds_tb = ds_tb.sel(time=slice(t0, t1))
    ds_tb = ds_tb.stack({"x": ("time", "view_ang", "i_obs")})

    # drop where tb from satellite is none (no footprint was close)
    ds_tb = ds_tb.sel(x=~ds_tb.tb_sat.isnull())

    # construct string ids
    df = pd.DataFrame(
        data=np.array(
            [
                ds_tb["instrument"].astype("str").values,
                ds_tb["satellite"].astype("str").values,
                ds_tb["channel"].astype("str").values,
                ds_tb["granule"].astype("str").values,
                ds_tb["footprint_id"].astype("int").astype("str").values,
            ]
        ).T
    )
    df = df[0] + "_" + df[1] + "_" + df[2] + "_" + df[3] + "_" + df[4]
    ds_tb["ix_sat"] = ("x", df.values)
    ds_tb = ds_tb.swap_dims({"x": "ix_sat"}).reset_coords()

    # get unique footprints
    fid, ix = np.unique(ds_tb.ix_sat, return_index=True)
    ds_tb = ds_tb.isel(ix_sat=ix)

    # keep only variables that are valid and do not come from aircraft
    sat_vars = [
        "tb_sat",
        "incidence_angle",
        "lon_sat",
        "lat_sat",
        "alt_sat",
        "scan_time",
        "footprint_id",
        "instrument",
        "satellite",
        "channel",
        "granule",
        "polarization",
        "n_if_offsets",
        "bandwidth",
        "center_freq",
        "if_offset_1",
        "if_offset_2",
        "avg_freq",
        "label",
        "label_pol",
        "i_obs",
    ]
    ds_tb = ds_tb[sat_vars]

    # replace string index by integer index (else interpolations do not work)
    ds_tb["ix_sat_int"] = ("ix_sat", np.arange(0, len(ds_tb.ix_sat)))
    ds_tb = ds_tb.swap_dims({"ix_sat": "ix_sat_int"})
    ds_tb = ds_tb.reset_coords()

    return ds_tb


def surftemp2sat(
    ds_tb, model, lon_var="lon_sat", lat_var="lat_sat", time_var="scan_time"
):
    """
    Get mean era-5 or esa surface temperature at an array of geographic
    locations.

    This script was initially developed for satellite footprints, but works
    also for aircraft flight tracks.

    Parameters
    ----------
    ds_tb: tb dataset with lon_sat and lat_sat variables
    model: model from which surface temperature is extracted (e.g. era-5)
    lon_var: longitude variable name
    lat_var: latitude variable name
    time_var: scan or measurement time variable name

    Returns
    -------
    da_ts: data array with surface temperatures in kelvins
    """

    # roi to reduce model sizes
    roi = [
        ds_tb[lon_var].min().item() - 1,
        ds_tb[lon_var].max().item() + 1,
        ds_tb[lat_var].min().item() - 1,
        ds_tb[lat_var].max().item() + 1,
    ]

    if model == "era-5":
        ds_mod = read_era5_single_levels(
            time=ds_tb[time_var].values[0], roi=roi
        )
        ds_mod = ds_mod.rename(
            {"longitude": "lon", "latitude": "lat", "skt": "ts"}
        )
        lon, lat = np.meshgrid(ds_mod.lon.values, ds_mod.lat.values)
        ds_mod = ds_mod.rename({"lon": "y", "lat": "x", "time": "mod_time"})

    elif model == "esa":
        ds_mod = read_esa_cmems(ds_tb[time_var].values[0], roi=roi)
        ds_mod = ds_mod.rename({"analysed_st": "ts"})
        lon, lat = np.meshgrid(ds_mod.lon.values, ds_mod.lat.values)
        ds_mod = ds_mod.rename({"lon": "y", "lat": "x", "time": "mod_time"})
    else:
        raise "Provided model must be era-5 or esa"

    # project geographic coordinates
    proj_in = ccrs.PlateCarree()
    proj_out = ccrs.epsg("3413")

    # get the closest observation to satellite footprint
    swath_shape = lon.shape
    swath = proj_out.transform_points(
        proj_in,
        lon.flatten(),
        lat.flatten(),
    )[:, :2]

    tree = KDTree(swath)

    footprint = proj_out.transform_points(
        proj_in,
        ds_tb[lon_var].values.flatten(),
        ds_tb[lat_var].values.flatten(),
    )[:, :2]

    distance, ix = tree.query(footprint)
    assert (np.max(distance) < 15000).all()
    ix_x, ix_y = np.unravel_index(ix, swath_shape)

    # get observations at indices
    ix_x = xr.DataArray(
        ix_x.reshape(ds_tb[lon_var].shape),
        dims=ds_tb[lon_var].dims,
        coords=ds_tb[lon_var].coords,
        name="x",
    )
    ix_y = xr.DataArray(
        ix_y.reshape(ds_tb[lon_var].shape),
        dims=ds_tb[lon_var].dims,
        coords=ds_tb[lon_var].coords,
        name="y",
    )

    ds_mod = ds_mod.isel(x=ix_x, y=ix_y)

    # temporal selection only for era-5, because others are daily products
    if model == "era-5":
        ds_mod = ds_mod.sel(mod_time=ds_tb[time_var], method="nearest")

    da_ts = ds_mod.ts.reset_coords(drop=True)

    return da_ts


def unique_footprint_ancillary():
    """
    Combines unique footprints with ancillary data and writes the output to a
    netcdf file. The output can be used for PAMTRA simulations of the satellite
    footprints that align with the flight track. Afterwards, the PAMTRA
    simulation/tb/emissivity can be aligned again with the flight track via
    the unique ix_sat id that gives every satellite pixel a unique id.
    """

    for flight_id in FLIGHT_IDS:
        print(flight_id)

        ds_tb = unique_footprints(flight_id=flight_id)

        if "AFLUX" in flight_id:
            model = "esa"
        else:
            model = "esa"
        ds_tb["ts"] = surftemp2sat(ds_tb=ds_tb, model=model)

        ds_tb["sea_ice"] = xr.full_like(ds_tb.lon_sat, fill_value=1)
        ds_tb["land"] = xr.full_like(ds_tb.lon_sat, fill_value=0)
        ds_tb["surf_alt"] = xr.full_like(ds_tb.lon_sat, fill_value=0)

        # write to file
        ds_tb.to_netcdf(
            os.path.join(
                os.environ["PATH_SEC"],
                "data/sea_ice_emissivity/sat_flight_track",
                f"{flight_id}_gpm_l1c_unique_ancil.nc",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "function",
        help="along-track dataset, unique footprint"
        ", or satellite footprint at aircraft time",
        type=str,
        choices=["atrack", "unique", "at_time"],
    )

    args = parser.parse_args()

    if args.function == "atrack":
        aircraft_track()
    elif args.function == "unique":
        unique_footprint_ancillary()
    elif args.function == "at_time":
        sat_footprint_at_time()
