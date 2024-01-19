"""
Emissivity calculation from pamtra simulations and observations for aircraft
and satellite footprint that match with flight track.
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
from lizard.readers.band_pass import read_band_pass, read_band_pass_combination
from lizard.readers.brightness_temperature import read_tb
from lizard.readers.sat_at_aircraft import read_sat_at_aircraft_unique

from si_emis.readers.pamtra_simulation import read_pamtra_simulation
from si_emis.retrieval.makesetups import read_setups
from si_emis.style import sensors
from si_emis.writers.emissivity import (
    write_emissivity_aircraft,
    write_emissivity_satellite,
)


def emissivity_equation(tb, tb_e0, tb_e1):
    """
    Emissivity calculation from observation and two radiative transfer
    simulations.

    References: e.g. Prigent et al. (1997), Mathew et al. (2008)

    Parameters
    ----------
    tb: observed brightness temperature
    tb_e0: brightness temperature under emissivity of 0
    tb_e1: brightness temperature under emissivity of 1

    Returns
    -------
    e: emissivity
    """

    e = (tb - tb_e0) / (tb_e1 - tb_e0)

    return e


def query_aircraft_setups(
    setups,
    kind,
    emissivity,
    ocean_refl,
    land_refl,
    sea_ice_refl,
    groundtemp,
    temp,
    relhum,
):
    """
    Queries the setup yamls for specific simulation parameters
    """

    return [
        setup
        for setup in setups.values()
        if setup["surface"]["kind"] == kind
        if setup["nmlSet"]["emissivity"] == emissivity
        if setup["surface"]["ocean_refl"] == ocean_refl
        if setup["surface"]["land_refl"] == land_refl
        if setup["surface"]["sea_ice_refl"] == sea_ice_refl
        if setup["offset"]["groundtemp"] == groundtemp
        if setup["offset"]["temp"] == temp
        if setup["offset"]["relhum"] == relhum
    ]


def query_satellite_setups(
    setups, kind, emissivity, ocean_refl, land_refl, sea_ice_refl
):
    """
    Queries the setup yamls for specific simulation parameters. Unlike for
    aircraft setups, these yamls do not contain the offset key.
    """

    return [
        setup
        for setup in setups.values()
        if setup["surface"]["kind"] == kind
        if setup["nmlSet"]["emissivity"] == emissivity
        if setup["surface"]["ocean_refl"] == ocean_refl
        if setup["surface"]["land_refl"] == land_refl
        if setup["surface"]["sea_ice_refl"] == sea_ice_refl
    ]


def get_offsets(setups, variables):
    """
    Returns all offset combinations in the setups file
    """

    offsets = []
    for name, setup in setups.items():
        offsets.append([setup["offset"][v] for v in variables])
    offsets = np.unique(offsets, axis=0)

    return offsets


def compare_dict(d1: dict, d2: dict, path: str = None, diff: dict = None):
    """
    Compares two nested dictionaries with the same keys and outputs keys where
    the values differ.
    """

    if diff is None:
        diff = {}
    if path is None:
        path = ""
    for k, v in d1.items():
        if isinstance(v, dict):
            compare_dict(v, d2[k], f"{path}/{k}", diff)
        elif d1[k] != d2[k]:
            diff.update({f"{path}/{k}": (d1[k], d2[k])})
    return diff


def get_offset_names(ds):
    """
    Returns the offset name of each index

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the offsets.
    """

    pos_dct = {}  # index for positive offsets
    neg_dct = {}  # index for negative offsets
    no_offset = None  # index for unperturbed variables
    for i_offset in ds.i_offset.values:
        offsets = ds.offset.sel(i_offset=i_offset).values
        offset = offsets[offsets != 0]
        if (offsets != 0).any():
            variable = ds.variable.sel(variable=offsets != 0).item()
            if offset > 0:
                pos_dct[variable] = i_offset
            else:
                neg_dct[variable] = i_offset
        else:
            no_offset = i_offset

    return pos_dct, neg_dct, no_offset


def error_propagation(ds):
    """
    Gaussian error propagation based on the perturbed radiative transfer
    simulations.
    The following sources of uncertainty are used:
    - uncertainty of brightness temperature observation (channel-dependent)
    - uncertainty of surface temperature
    - uncertainty of atmospheric temperature
    - uncertainty of relative humidity

    Important note:
    The perturbed radiative transfer simulations are performed for +/- the
    assumed uncertainty of Ta, Ts, and RH.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the perturbed radiative transfer simulations,
        tb uncertainty, and offsets.

    Returns
    -------
    ds : xr.Dataset
        Dataset containing the propagated uncertainty of the emissivity
        estimate (e_unc, e_rel_unc).
    """

    # assumed tb uncertainty depending on the channel
    ds["tb_unc"] = xr.DataArray(
        np.array([2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        dims="channel",
        coords={"channel": ds.channel},
        attrs=dict(
            standard_name="brightness_temperature_uncertainty",
            long_name="brightness temperature uncertainty",
            units="K",
            description="uncertainty of brightness temperature",
        ),
    )

    # get the perturbed variables and the sign of the perturbation
    pos_dct, neg_dct, no_offset = get_offset_names(ds)

    # emissivity changes from perturbed radiative transfer simulations
    de_ts = (
        ds["e"].sel(i_offset=pos_dct["groundtemp"])
        - ds["e"].sel(i_offset=neg_dct["groundtemp"])
    ) / 2
    de_ta = (
        ds["e"].sel(i_offset=pos_dct["temp"])
        - ds["e"].sel(i_offset=neg_dct["temp"])
    ) / 2
    de_rh = (
        ds["e"].sel(i_offset=pos_dct["relhum"])
        - ds["e"].sel(i_offset=neg_dct["relhum"])
    ) / 2

    # perturbation magnitudes
    dts = (
        (
            ds["offset"].sel(
                i_offset=pos_dct["groundtemp"], variable="groundtemp"
            )
            - ds["offset"].sel(
                i_offset=neg_dct["groundtemp"], variable="groundtemp"
            )
        ).item()
    ) / 2
    dta = (
        (
            ds["offset"].sel(i_offset=pos_dct["temp"], variable="temp")
            - ds["offset"].sel(i_offset=neg_dct["temp"], variable="temp")
        ).item()
    ) / 2
    drh = (
        (
            ds["offset"].sel(i_offset=pos_dct["relhum"], variable="relhum")
            - ds["offset"].sel(i_offset=neg_dct["relhum"], variable="relhum")
        ).item()
    ) / 2

    # emissivity sensitivity to tb, ts, ta, rh
    de_dtb = 1 / (
        ds["tb_e1"].sel(i_offset=no_offset)
        - ds["tb_e0"].sel(i_offset=no_offset)
    )
    de_dts = de_ts / dts
    de_dta = de_ta / dta
    de_drh = de_rh / drh

    # error propagation equation
    ds["e_unc"] = np.sqrt(
        de_dtb**2 * ds["tb_unc"] ** 2
        + de_dts**2 * dts**2
        + de_dta**2 * dta**2
        + de_drh**2 * drh**2
    )

    # relative uncertainty of emissivity
    ds["e_rel_unc"] = ds["e_unc"] / ds["e"].sel(i_offset=no_offset) * 100

    return ds


def prepare_pamtra(file0, file1, ds_tb, da_avg_freq, axis, angle):
    """
    Reads and prepares PAMTRA simulations with emissivity of 0 and 1 for either
    emissivity retrievals (given a certain observed tb), or tb retrievals
    given a certain emissivity or in combination with a specularity parameter.

    If files are not found, the function returns None instead of raising an
    exception!

    Parameters
    ----------
    file0: pamtra simulation file with emissivity of 0
    file1: pamtra simulation file with emissivity of 1
    ds_tb: tb dataset with ancillary data on incidence angle
    da_avg_freq: frequencies that will be averaged for each
      channel to map the frequency coordinate in pamtra to channel coordinate
      of the tb dataset
    axis: variable name that will be swapped with the default 'grid_x' (e.g.
      time or another variable that is contained in the pamtra simulation
      dataset)
    angle: name of incidence angle variable in tb dataset ds_tb

    Returns
    -------
    ds_pam0: pamtra simulation under emissivity of 0
    ds_pam1: pamtra simulation under emissivity of 1
    """

    try:
        ds_pam0 = read_pamtra_simulation(file0)
        ds_pam1 = read_pamtra_simulation(file1)

    except FileNotFoundError:
        print(f"Could not find PAMTRA simulation files {file0} and/or {file1}")
        print("This function will return None for both files")
        return None, None

    # make sure that the simulations only differ in emissivity
    comp = compare_dict(ds_pam0.attrs, ds_pam1.attrs)
    assert list(comp.keys()) == ["/created", "/pyPamtra_settings/emissivity"]
    assert comp["/pyPamtra_settings/emissivity"] == ("0", "1")

    # restructure pamtra dataset
    ds_pam0 = ds_pam0.sel(
        polarization="V", outlevel=1, grid_y=0, direction="up"
    ).reset_coords(drop=True)
    ds_pam1 = ds_pam1.sel(
        polarization="V", outlevel=1, grid_y=0, direction="up"
    ).reset_coords(drop=True)
    ds_pam0 = ds_pam0.swap_dims({"grid_x": axis}).reset_coords(drop=True)
    ds_pam1 = ds_pam1.swap_dims({"grid_x": axis}).reset_coords(drop=True)

    # align dataset along axis
    ds_tb, ds_pam0, ds_pam1 = xr.align(ds_tb, ds_pam0, ds_pam1)

    # average frequencies
    ds_pam0 = ds_pam0.sel(frequency=da_avg_freq).mean("n_avg_freq")
    ds_pam1 = ds_pam1.sel(frequency=da_avg_freq).mean("n_avg_freq")

    # interpolate angle
    ds_pam0 = ds_pam0.interp(
        angle=ds_tb[angle], method="linear"
    ).reset_coords()
    ds_pam1 = ds_pam1.interp(
        angle=ds_tb[angle], method="linear"
    ).reset_coords()

    return ds_pam0, ds_pam1


def compute_emissivity(ds_tb, tb, file0, file1, da_avg_freq, axis, angle):
    """
    Computes emissivity from two pamtra simulations and the observed
    brightness temperature dataset

    Parameters
    ----------
    file0: pamtra simulation file with emissivity of 0
    file1: pamtra simulation file with emissivity of 1
    ds_tb: tb dataset with ancillary data on incidence angle
    da_avg_freq: frequencies that will be averaged for each
      channel to map the frequency coordinate in pamtra to channel coordinate
      of the tb dataset
    axis: variable name that will be swapped with the default 'grid_x' (e.g.
      time or another variable that is contained in the pamtra simulation
      dataset)
    angle: name of incidence angle variable in tb dataset ds_tb
    tb: name of tb variable in tb dataset ds_tb

    Returns
    -------
    da_e: emissivity data array
    da_dtb: tb sensitivity to changes in emissivity
    """

    # prepare pamtra data
    ds_pam0, ds_pam1 = prepare_pamtra(
        file0, file1, ds_tb, da_avg_freq, axis, angle
    )

    # this makes sure that if a file is not there, the calculation continues
    if ds_pam0 is None and ds_pam1 is None:
        ds_tb["tb_e0"] = np.nan
        ds_tb["tb_e1"] = np.nan

    else:
        ds_tb["tb_e0"] = ds_pam0["tb"]
        ds_tb["tb_e1"] = ds_pam1["tb"]

    ds_tb["e"] = emissivity_equation(
        tb=ds_tb[tb], tb_e0=ds_tb["tb_e0"], tb_e1=ds_tb["tb_e1"]
    )
    ds_tb["dtb"] = ds_tb["tb_e1"] - ds_tb["tb_e0"]

    return ds_tb


def emissivity_aircraft(flight_id):
    """
    Computes emissivity for a list of flight ids
    """

    mission, platform, name = flight_id.split("_")

    ds_tb = read_tb(flight_id)

    sen = sensors[mission][platform]
    if sen["sensor2"] is None:
        ds_bp = read_band_pass(sen["sensor1"])  # hamp on halo
    else:
        ds_bp = read_band_pass_combination(**sen)  # polar 5

    file = f"./src/si_emis/retrieval/setups/setup_{flight_id}.yaml"
    setups = read_setups(file)

    # offset variables and corresponding offsets in setup files
    variables = ["groundtemp", "temp", "relhum"]
    offsets = get_offsets(setups, variables)

    # compute emissivity and sensitivity for each idealized setup
    lst_ds = []
    lst_sfc_refl = []
    lst_offset_groundtemp = []
    lst_offset_temp = []
    lst_offset_relhum = []
    lst_i_offset = []
    for surf_refl in ["L", "S"]:
        for i, offset in enumerate(offsets):
            offset_dct = {v: o for v, o in zip(variables, offset)}
            kwargs = dict(
                setups=setups,
                kind="idealized",
                ocean_refl=surf_refl,
                land_refl=surf_refl,
                sea_ice_refl=surf_refl,
                **offset_dct,
            )

            # get setup files with emissivity 0 and 1
            setup_e0 = query_aircraft_setups(emissivity=0, **kwargs)[0]
            setup_e1 = query_aircraft_setups(emissivity=1, **kwargs)[0]

            files = dict(
                file0=setup_e0["general"]["outfile"],
                file1=setup_e1["general"]["outfile"],
            )

            # compute emissivity and tb sensitivity to emissivity changes
            ds = compute_emissivity(
                ds_tb=ds_tb,
                da_avg_freq=ds_bp.avg_freq,
                angle="ac_zen",
                tb="tb",
                axis="time",
                **files,
            )

            # store dataset and coordinates of computation
            lst_ds.append(ds.copy(deep=True))
            lst_sfc_refl.append(surf_refl)
            lst_offset_groundtemp.append(offset_dct["groundtemp"])
            lst_offset_temp.append(offset_dct["temp"])
            lst_offset_relhum.append(offset_dct["relhum"])
            lst_i_offset.append(i)

    # merge output into dataset
    ds = xr.concat(lst_ds, dim="c", data_vars=["e", "dtb", "tb_e0", "tb_e1"])
    ds.coords["surf_refl"] = ("c", lst_sfc_refl)
    ds.coords["i_offset"] = ("c", lst_i_offset)
    ds = ds.set_index(c=["surf_refl", "i_offset"]).unstack("c")
    ds.coords["variable"] = variables
    ds["offset"] = (("i_offset", "variable"), offsets)

    # set emissivity to nan where surface temperature is nan. in these cases,
    # pamtra used the lowest atmospheric temperature instead.
    is_measured_ts = ds["ts"].notnull()
    ds["e"] = ds["e"].where(is_measured_ts)
    ds["tb_e0"] = ds["tb_e0"].where(is_measured_ts)
    ds["tb_e1"] = ds["tb_e1"].where(is_measured_ts)
    ds["dtb"] = ds["dtb"].where(is_measured_ts)

    # emissivity uncertainty in percent from error propagation
    ds = error_propagation(ds)

    # add attributes
    ds["tb_e0"].attrs = dict(
        standard_name="brightness_temperature",
        long_name="brightness temperature under emissivity of 0",
        units="K",
        description="brightness temperature simulated with PAMTRA under emissivity of 0",
    )

    ds["tb_e1"].attrs = dict(
        standard_name="brightness_temperature",
        long_name="brightness temperature under emissivity of 1",
        units="K",
        description="brightness temperature simulated with PAMTRA under emissivity of 1",
    )

    ds["e"].attrs = dict(
        standard_name="emissivity",
        long_name="emissivity",
        description="emissivity calculated from observation and PAMTRA simulation",
    )

    ds["surf_refl"].attrs = dict(
        standard_name="surface_reflection_type",
        long_name="surface reflection type",
        description="surface reflection type (L: Lambertian, S: specular)",
    )

    ds["i_offset"].attrs = dict(
        standard_name="offset_index",
        long_name="offset index",
        description="index of offset combination for perturbed radiative transfer simulations",
    )

    ds["variable"].attrs = dict(
        standard_name="offset_variable",
        long_name="offset variable",
        description="variable on which offset is applied during radiative transfer simulations",
    )

    ds["offset"].attrs = dict(
        standard_name="offset",
        long_name="offset",
        description="offset applied during radiative transfer simulations. This in in Kelvin "
        "for temperature and ground temperature, and in percent for relative humidity",
    )

    ds["dtb"].attrs = dict(
        standard_name="brightness_temperature_sensitivity",
        long_name="brightness temperature sensitivity",
        units="K",
        description="difference in brightness temperature between emissivity of 1 and 0",
    )

    ds["e_unc"].attrs = dict(
        standard_name="emissivity_uncertainty",
        long_name="emissivity uncertainty",
        description="emissivity uncertainty calculated "
        "from Gaussian error propagation with uncertainties from "
        "brightness temperature, surface temperature, atmospheric "
        "temperature, and relative humidity.",
    )

    ds["e_rel_unc"].attrs = dict(
        standard_name="relative_emissivity_uncertainty",
        long_name="relative emissivity uncertainty",
        units="%",
        description="relative emissivity uncertainty calculated "
        "from Gaussian error propagation with uncertainties from "
        "brightness temperature, surface temperature, atmospheric "
        "temperature, and relative humidity.",
    )

    # add global attributes
    ds.attrs = dict(
        title=(
            "Surface microwave emissivity and uncertainty during "
            f"{mission} {name} from MiRAC onboard Polar 5"
        ),
        description=(
            "Emissivity and uncertainty calculated from observed TB and "
            "PAMTRA simulations. Additional data on aircraft and footprint "
            "location are provided as well. Note that the 89 GHz channel "
            "(channel 1) is inclined by 25 degrees, and all other channels by "
            "0 degrees (nadir)"
        ),
        data_quality=(
            "The emissivity is subject to potential uncertainties due to "
            "TB measurement errors, "
            "unknown sub-surface temperature, and "
            "uncertainties in the atmospheric temperature and humidity "
            "profiles. "
            "These might cause significant emissivity biases. "
            "Especially observations of the channels 2 to 6 towards the "
            "center of the 183.31 GHz absorption line might be subject to "
            "higher uncertainties. Please check the emissivity uncertainty "
            r"to assess the quality of the retrieval. Values above 10% can be "
            "considered as unreliable. Also, the dtb variable provides a hint "
            "on uncertainties, where values below 50 K can be considered as "
            "unreliable. "
            "Users are advised to refer to the associated reference for a "
            "comprehensive description of data quality."
        ),
        source=(
            "Derived from airborne observations and radiative transfer "
            "simulations with the Passive and Active Microwave Radiative "
            "Transfer (PAMTRA) model (Mech et al., 2020). See the "
            "manuscript for details (Risse et al., 2024)."
        ),
        history=(
            "Calculated from observed TB and PAMTRA simulations with "
            "additional information on the atmospheric profile. Data "
            "underwent preprocessing steps, including footprint position "
            "calculation, TB offset bias correction at 89 GHz, and infrared "
            "TB conversion to surface temperature to derive surface microwave "
            "emissivity and uncertainty"
        ),
        project="Arctic Amplification (AC3)",
        mission=mission,
        platform="Polar 5",
        flight_id=name,
        instrument=(
            "Microwave Radar/Radiometer for Arctic Clouds - active "
            "(MiRAC-A) and - passive (MiRAC-P)"
        ),
        institution=(
            "Institute for Geophysics and Meteorology, "
            "University of Cologne, Cologne, Germany"
        ),
        author="Nils Risse",
        contact="n.risse@uni-koeln.de",
        references="Risse et al. (2024; submitted to The Cryosphere)",
        created=str(np.datetime64("now")),
        convention="CF-1.8",
        featureType="trajectory",
    )

    write_emissivity_aircraft(ds, flight_id)


def emissivity_satellite(flight_id):
    """
    Compute emissivity for satellite footprints that align with the flight
    track.
    """

    ds_tb = read_sat_at_aircraft_unique(flight_id)

    file = f"./src/si_emis/retrieval/setups/setup_sat_{flight_id}.yaml"
    setups = read_setups(file)

    # compute emissivity for each setup
    lst_ds = []
    lst_sfc_refl = []
    for surf_refl in ["L", "S"]:
        kwargs = dict(
            setups=setups,
            kind="idealized",
            ocean_refl=surf_refl,
            land_refl=surf_refl,
            sea_ice_refl=surf_refl,
        )

        # get setup files with emissivity 0 and 1
        setup_e0 = query_satellite_setups(emissivity=0, **kwargs)[0]
        setup_e1 = query_satellite_setups(emissivity=1, **kwargs)[0]

        files = dict(
            file0=setup_e0["general"]["outfile"],
            file1=setup_e1["general"]["outfile"],
        )

        # compute emissivity and tb sensitivity to emissivity changes
        ds = compute_emissivity(
            ds_tb=ds_tb,
            da_avg_freq=ds_tb.avg_freq,
            angle="incidence_angle",
            tb="tb_sat",
            axis="ix_sat_int",
            **files,
        )

        # store dataset and coordinates of computation
        lst_ds.append(ds.copy(deep=True))
        lst_sfc_refl.append(surf_refl)

    # merge output into dataset
    ds = xr.concat(
        lst_ds,
        dim=pd.Index(lst_sfc_refl, name="surf_refl"),
        data_vars=["e", "dtb", "tb_e0", "tb_e1"],
    )

    # add attributes
    ds["dtb"].attrs = dict(
        units="K",
        description="difference in brightness temperature between emissivity of 1 and 0",
    )

    # add global attributes
    ds.attrs = dict(
        title=f"Surface emissivity and uncertainty of satellites during "
        f"{flight_id}",
        research_flight=flight_id,
        history="Calculated from observed TB and PAMTRA simulations",
        author="Nils Risse",
        contact="n.risse@uni-koeln.de",
        created=str(np.datetime64("now")),
    )

    write_emissivity_satellite(ds, flight_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        flight_ids = [
            "ACLOUD_P5_RF23",
            "ACLOUD_P5_RF25",
            "AFLUX_P5_RF08",
            "AFLUX_P5_RF14",
            "AFLUX_P5_RF15",
        ]
        for flight_id in flight_ids:
            print(flight_id)
            emissivity_aircraft(flight_id)
            emissivity_satellite(flight_id)

    else:
        mode = sys.argv[1]
        flight_ids = sys.argv[2:]

        if mode == "aircraft":
            for flight_id in flight_ids:
                print(flight_id)
                emissivity_aircraft(flight_id)
        elif mode == "satellite":
            for flight_id in flight_ids:
                print(flight_id)
                emissivity_satellite(flight_id)
        else:
            print("Invalid mode. Use 'aircraft' or 'satellite'.")
            sys.exit(1)
