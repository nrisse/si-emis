"""
This module combines airborne and satellite footprints.

It creates in the end one file per research flight given the distance and hour
threshold.
"""

import os

import numpy as np
import xarray as xr
from lizard.bitflags import read_flag_xarray

from si_emis.data_preparation.surface import land_ocean_shore_mask
from si_emis.readers.emissivity import (
    read_emissivity_aircraft,
    read_emissivity_satellite,
)

DTB_THRESHOLD = 40  # surface sensitivity threshold in K
HOUR_THRESHOLD = 2  # time difference threshold in hours


def airborne_filter(
    ds,
    channel_var="channel",
    surf_type="ocean",
    exclude_shore=True,
    drop_times=False,
    dtb_filter=True,
    angle_filter=True,
    dtb_keep_tb=False,
):
    """
    Filter airborne emissivity dataset.

    Filters:
        - 89 GHz (channel 1) aircraft range must be above 500 m
        - roll angles at 89 GHz must be smaller than 5deg

    Parameters
    ----------
    ds: airborne emissivity dataset
    channel_var: variable name of airborne channels
    surf_type: choose surface type (ocean, ocean+ice, ocean-ice, land,
       glacier).
       Note that the sea ice mask is coarse. If ocean is definitely covered by
       sea ice and sea ice should be kept, then use 'ocean'. Default: ocean
    exclude_shore: whether to exclude buffer close to shore. Default: True
    drop_times: drops times, where e is filtered out at every channel
    dtb_filter: sets values to nan where surface sensitivity is below 0.1
    angle_filter: whether to filter for high incidence angles for near-nadir
      observations.
    dtb_filter_tb: whether to filter the Tb values also with the dTb threshold.
      This might not be wanted if also TB from inner 183 GHz channels are
      plotted. Default is False.

    Returns
    -------
    ds: filtered airborne emissivity dataset
    """

    if not dtb_filter:
        Warning(
            "Not applying dTB filter. The output include crude emissivity"
            "values"
        )

    # variables on which the filtering is applied
    v = [
        "tb",
        "e",
        "ts",
        "tb_e0",
        "tb_e1",
        "dtb",
        "e_unc",
    ]

    # distance to surface
    ds[v].loc[{channel_var: 1}] = (
        ds[v]
        .loc[{channel_var: 1}]
        .where(ds.ac_range.sel({channel_var: 1}) > 500)
    )

    if angle_filter:
        # roll angle
        ds[v] = ds[v].where((ds.ac_roll > -10) & (ds.ac_roll < 10))

        # pitch angle
        ds[v] = ds[v].where((ds.ac_pitch > -10) & (ds.ac_pitch < 10))

        # incidence angle
        ds[v].loc[{channel_var: slice(2, 9)}] = (
            ds[v]
            .loc[{channel_var: slice(2, 9)}]
            .where(ds.ac_zen.sel({channel_var: slice(2, 9)}) < 10)
        )

    # select sea ice pixels and remove land pixels
    ds = read_flag_xarray(ds, "surf_type", drop=True)

    if surf_type == "ocean":
        ds[v] = ds[v].where(ds.ocean == 1)

    elif surf_type == "ocean+ice":
        ds[v] = ds[v].where((ds.ocean == 1) & (ds.sea_ice == 1))

    elif surf_type == "ocean-ice":
        ds[v] = ds[v].where((ds.ocean == 1) & (ds.sea_ice == 0))

    elif surf_type == "land":
        ds[v] = ds[v].where(ds.land == 1)

    elif surf_type == "glacier":
        ds[v] = ds[v].where(ds.glacier == 1)

    # pre-defined buffer around shore line
    if exclude_shore:
        ds[v] = ds[v].where(ds.shore == 0)

    # sensitivity to surface must be above threshold (note that this depends on
    # the surface reflection type and therefore tb and ts depend on it now as
    # well, although they differ only in the occurrence of nan values)
    if dtb_keep_tb:
        v.remove("tb")

    if dtb_filter:
        ds[v] = ds[v].where(ds.dtb >= DTB_THRESHOLD)

    if drop_times:
        drop_along = list(ds.e.dims)
        drop_along.remove("time")
        ds = ds.sel(time=~ds.e.isnull().all(drop_along))

    return ds


def airborne_emissivity_merged(
    flight_ids=None, filter_kwargs=None, without_offsets=True
):
    """
    Reads airborne emissivity for a list of flights by concatenation along
    the time dimension. The research flight id is provided as a function of
    time.

    Parameters
    ----------
    flight_ids: flight ids to be concatenated
    filter_kwargs: keyword arguments for airborne filter function
    without_offsets: whether to read emissivity and emissivity sensitivity
      calculated under the perturbed profiles
    """

    if filter_kwargs is None:
        filter_kwargs = dict()

    if flight_ids is None:
        flight_ids = [
            "ACLOUD_P5_RF23",
            "ACLOUD_P5_RF25",
            "AFLUX_P5_RF07",
            "AFLUX_P5_RF08",
            "AFLUX_P5_RF13",
            "AFLUX_P5_RF14",
            "AFLUX_P5_RF15",
        ]

    lst_ds = []
    lst_flight_id = []
    lst_i_flight_id = []
    dct_flight_id = {}
    for i, flight_id in enumerate(flight_ids):
        dct_flight_id[i] = flight_id
        ds = read_emissivity_aircraft(
            flight_id, without_offsets=without_offsets
        )
        ds = airborne_filter(ds, **filter_kwargs)
        lst_ds.append(ds)
        lst_flight_id.extend(np.full(len(ds.time), fill_value=flight_id))
        lst_i_flight_id.extend(np.full(len(ds.time), fill_value=i))

    # concatenate along time dimension
    data_vars = list(lst_ds[0])
    data_vars.remove("center_freq")
    data_vars.remove("n_if_offsets")
    data_vars.remove("if_offset_1")
    data_vars.remove("bandwidth")
    data_vars.remove("polarization")
    data_vars.remove("view_ang")
    ds_merged = xr.concat(lst_ds, dim="time", data_vars=data_vars)
    ds_merged["flight_id"] = ("time", lst_flight_id)
    ds_merged["i_flight_id"] = ("time", lst_i_flight_id)

    return ds_merged, dct_flight_id


def count_checking(ds_es, surf_refl):
    """
    Only those satellite footprints are kept for which at least a certain
    number of airborne emissivity values exist. For 89 GHz satellite channels,
    the count from MiRAC-A 89 GHz is used. For the higher frequency satellite
    channels, the count from MiRAC-P 243 GHz is used. Hence, if Polar 5 flies
    below about 500 m, no satellite 89 GHz is available.

    The count threshold depends on the instrument.
    For AMSR2 it is 17 and for MHS, AMTS, and SSMIS it is 50.
    Reason: 1/3 of the footprint size should be at least covered
    Assumption: aircraft footprint every 100 m
    AMSR2 footprint is about 5 km: 1.7 km.
    MHS, AMTS, and SSMIS instruments have 15 km footprint: 5 km.
    We neglect here that ATMS 89 GHz has 31 km resolution, which is acceptable
    as we do not consider the sampling rate of the satellite.

    Parameters
    ----------
    ds_es: emissivity from satellite
    surf_refl: surface reflection

    Returns
    -------
    ds_es: xr.Dataset
        satellite emissivity dataset
    """

    count_amsr2 = 17
    count_other = 50

    cnt_89 = ds_es["ea_e_count"].sel(surf_refl=surf_refl, ac_channel=1)
    cnt_243 = ds_es["ea_e_count"].sel(surf_refl=surf_refl, ac_channel=8)

    cnt_89 = cnt_89.reset_coords(drop=True)
    cnt_243 = cnt_243.reset_coords(drop=True)

    # keep only aircraft-satellite pairs with a high number of footprints
    keep_footprint = (
        (ds_es.center_freq < 100)
        & (ds_es.instrument == "AMSR2")
        & (cnt_89 >= count_amsr2)
        | (ds_es.center_freq < 100)
        & (ds_es.instrument != "AMSR2")
        & (cnt_89 >= count_other)
        | (ds_es.center_freq > 100) & (cnt_243 >= count_other)
    )

    ds_es = ds_es.sel(ix_sat_int=keep_footprint)

    return ds_es


def downsampling(ds_es, ds_ea, ds_sc2ac):
    """
    Downsample airborne emissivity to satellite footprints.

    Parameters
    ----------
    ds_es: xr.Dataset
        satellite emissivity dataset
    ds_ea: xr.Dataset
        airborne emissivity dataset
    ds_sc2ac: xr.Dataset
        match between airborne and spaceborne dataset

    Returns
    -------
    ds_es: xr.Dataset
        satellite emissivity dataset
    """

    # calcualte mean surface temperature
    ds_es["ea_ts_mean"] = ds_ea["ts"].where(ds_sc2ac["at_time"]).mean("time")

    # calculate mean tb (this is the measured tb and included altitude changes)
    ds_es["ea_tb_mean"] = ds_ea["tb"].where(ds_sc2ac["at_time"]).mean("time")

    # calculate number of airborne emissivity samples within footprint
    ds_es["ea_e_count"] = ds_ea["e"].where(ds_sc2ac["at_time"]).count("time")

    # calculate standard deviation of airborne emissivity within footprint
    ds_es["ea_e_std"] = ds_ea["e"].where(ds_sc2ac["at_time"]).std("time")

    # calculate iqr of airborne emissivity within footprint
    ds_es["ea_e_iqr"] = ds_ea["e"].where(ds_sc2ac["at_time"]).quantile(
        0.75, dim="time"
    ) - ds_ea["e"].where(ds_sc2ac["at_time"]).quantile(0.25, dim="time")

    # calculate mean uncertainty of airborne emissivity within footprint
    ds_es["ea_e_unc"] = ds_ea["e_unc"].where(ds_sc2ac["at_time"]).mean("time")

    # calculate mean relative uncertainty of airborne emissivity within footprint
    ds_es["e_rel_unc"] = (
        ds_ea["e_rel_unc"].where(ds_sc2ac["at_time"]).mean("time")
    )

    # calculate mean emissivity within footprint
    # note that this is not the same as averaging the emissivity, it needs to
    # be derived from the mean surface temperature and the mean brightness
    # temperature. The mean Tb is simply the surface temperature times the
    # emissivity for every time step (then not atmosphere contribution needed)
    # e*ts / mean surface temperature
    ds_es["ea_e_mean"] = (
        ds_ea["e"].where(ds_sc2ac["at_time"]).mean("time")
        * ds_ea["ts"].where(ds_sc2ac["at_time"]).mean("time")
        / ds_es["ea_ts_mean"]
    )

    # compute all variables
    ds_es.compute()

    return ds_es


def downsampling_filters(ds_es, ds_ea, ds_sc2ac, hour_thr):
    """
    Downsample aircraft emissivity to satellite footprint. Several filters
    can be applied such as temporal filter and distance filter.

    The distance threshold depends on the instrument. For AMSR2 it is 2.5 km
    from the footprint center and for the other instruments it is 8 km.
    Only airborne footprints within this distance are considered for the
    comparison.

    Parameters
    ----------
    ds_es: satellite emissivity dataset
    ds_ea: airborne emissivity dataset
    ds_sc2ac: match between airborne and spaceborne dataset
    hour_thr: maximum temporal offset between airborne time and satellite
      scan time

    Returns
    -------
    ds_sc2ac: with updated 'at_time' variable
    """

    dist_max_amsr2 = 2500  # meters
    dist_max_other = 8000  # meters

    hour_thr = np.timedelta64(hour_thr, "h")
    time_filter = np.abs(ds_es.scan_time - ds_ea.time) < hour_thr
    distance_filter = (ds_es.instrument == "AMSR2") & (
        ds_sc2ac.distance <= dist_max_amsr2
    ) | (ds_es.instrument != "AMSR2") & (ds_sc2ac.distance <= dist_max_other)

    # merge filters
    ds_sc2ac["at_time"] = ds_sc2ac["at_time"] * time_filter * distance_filter

    return ds_sc2ac


def satellite_filter(ds_es, dtb_filter=True):
    """
    Filters satellite emissivity dataset near aircraft flight track.

    Parameters
    ----------
    ds_es : xr.Dataset
        satellite emissivity dataset. Coordinate: ix_sat_int. This is the
        index of the satellite footprint. Hence, this filtering drops entire
        satellite footprints along this coordinate.
    dtb_filter: bool, optional
        whether to filter satellite observations for surface sensitivity.
        This is done by default. An application for not applying the filter
        is the comparison of brightness temperatures from the satellite
        with PAMTRA forward simulation specifically for pixels that are
        collocated with airborne observations.
    """

    # variables on which the filtering is applied
    v = ["tb_sat", "e", "ts", "tb_e0", "tb_e1", "dtb"]

    # sensitivity to surface must be above threshold
    if dtb_filter:
        ds_es[v] = ds_es[v].where(ds_es.dtb >= DTB_THRESHOLD)

    # quality check
    ds_es[v] = ds_es[v].where(ds_es.tb_sat < 300)

    # keep only open ocean
    ix_amsr2 = ds_es.instrument == "AMSR2"
    ix_other = ds_es.instrument != "AMSR2"
    d_amsr2 = 2500  # min distance of footprint center to shore line in meters
    d_other = 8000
    mask_amsr2 = land_ocean_shore_mask(
        lon=ds_es.lon_sat.sel(ix_sat_int=ix_amsr2),
        lat=ds_es.lat_sat.sel(ix_sat_int=ix_amsr2),
        buffer=d_amsr2,
    )
    mask_other = land_ocean_shore_mask(
        lon=ds_es.lon_sat.sel(ix_sat_int=ix_other),
        lat=ds_es.lat_sat.sel(ix_sat_int=ix_other),
        buffer=d_other,
    )

    land_mask_amsr2, ocean_mask_amsr2, shore_mask_amsr2 = mask_amsr2
    land_mask_other, ocean_mask_other, shore_mask_other = mask_other

    mask_amsr2 = (
        (ocean_mask_amsr2 == 1)
        & (shore_mask_amsr2 == 0)
        & (land_mask_amsr2 == 0)
    )
    mask_other = (
        (ocean_mask_other == 1)
        & (shore_mask_other == 0)
        & (land_mask_other == 0)
    )
    mask = xr.concat([mask_amsr2, mask_other], dim="ix_sat_int")
    mask = mask.sortby("ix_sat_int")

    ds_es = ds_es.sel(ix_sat_int=(mask == 1))

    return ds_es


def main(
    flight_id: str,
    hour_thr: float,
    dtb_filter_sat: bool = True,
):
    """
    Prepares input data for specific research flight

    Parameters
    ----------
    flight_id: str
        ac3airborne flight id
    hour_thr: float
        time threshold between satellite and aircraft in hours
    dtb_filter_sat: bool, optional
        whether to filter satellite observations for surface sensitivity.
        This is done by default. An application for not applying the filter
        is the comparison of brightness temperatures from the satellite
        with PAMTRA forward simulation specifically for pixels that are
        collocated with airborne observations.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset, xr.Dataset]
        A tuple containing three datasets: ds_ea, ds_es, and ds_sc2ac.
    """

    print(f"Processing {flight_id}")

    # read airborne and spaceborne emissivity datasets
    ds_ea = read_emissivity_aircraft(flight_id)
    ds_ea = ds_ea.rename({"channel": "ac_channel"})
    ds_es = read_emissivity_satellite(flight_id)
    ds_sc2ac = xr.open_dataset(
        os.path.join(
            os.environ["PATH_SEC"],
            "data/sea_ice_emissivity/sat_flight_track",
            f"{flight_id}_gpm_l1_sc2ac.nc",
        )
    )

    ds_sc2ac = ds_sc2ac.chunk({"ix_sat_int": 1000})
    ds_ea = ds_ea.chunk({"ac_channel": 1})
    ds_es = ds_es.chunk({"ix_sat_int": 1000})

    # ensure that the ix_sat_int in emissivity dataset is same as in the mask
    ds_sc2ac = ds_sc2ac.swap_dims({"ix_sat_int": "ix_sat"})
    ds_sc2ac = ds_sc2ac.sel(ix_sat=ds_es.ix_sat.values)  # reorder
    ds_sc2ac["ix_sat_int"] = ("ix_sat", ds_es.ix_sat_int.values)
    ds_sc2ac = ds_sc2ac.swap_dims({"ix_sat": "ix_sat_int"})

    assert (ds_es.instrument == ds_sc2ac.instrument).all()
    assert (ds_es.satellite == ds_sc2ac.satellite).all()
    assert (ds_es.channel == ds_sc2ac.channel).all()
    assert (ds_es.granule == ds_sc2ac.granule).all()
    assert (ds_es.footprint_id == ds_sc2ac.footprint_id).all()
    assert (ds_es.ix_sat_int == ds_sc2ac.ix_sat_int).all()
    assert (ds_es.ix_sat == ds_sc2ac.ix_sat).all()

    ds_ea, ds_sc2ac = xr.align(ds_ea, ds_sc2ac)
    ds_sc2ac = ds_sc2ac.sel(view_ang=ds_ea.view_ang).reset_coords(drop=True)

    # filter airborne dataset
    print("Filtering airborne dataset")
    ds_ea = airborne_filter(ds_ea, channel_var="ac_channel")

    # filter satellite dataset
    print("Filtering satellite dataset")
    ds_es = satellite_filter(ds_es, dtb_filter=dtb_filter_sat)

    # filter by time offset and distance
    print("Filtering by time offset and distance")
    ds_sc2ac = downsampling_filters(ds_es, ds_ea, ds_sc2ac, hour_thr=hour_thr)

    # downsample airborne emissivity, tb, and ts
    print("Downsampling")
    ds_es = downsampling(ds_es, ds_ea, ds_sc2ac)

    # remove satellite footprints with low number of airborne footprints
    print("Count checking")
    ds_es = count_checking(
        ds_es,
        surf_refl="L",
    )

    # write outfile
    file = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_emissivity/airsat",
        f"{flight_id}_airsat.nc",
    )
    print(f"Writing {file}")
    ds_es.to_netcdf(file)


if __name__ == "__main__":
    # prepare data for every research flight and save result to file
    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]

    for f in flight_ids:
        main(
            flight_id=f,
            hour_thr=HOUR_THRESHOLD,
            dtb_filter_sat=True,
        )
