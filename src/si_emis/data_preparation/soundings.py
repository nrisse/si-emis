"""
Format conversion of PANGAEA radiosonde data at Ny-Alesund.

Conversions:
- distinguish individual soundings and write them to NetCDF.
- merge soundings onto a common vertical grid to have all soundings in one 
file. 
- all soundings in one file
- times can occur more than once and are not a real time index, but just
the time of a sounding
- consecutive soundings can be distinguished by etim: time ellapsed since
launch
- if delta etim is negative, a new sounding starts
- then this segment can be saved as single sounding

Data citation:
Maturilli, M (2020): High resolution radiosonde measurements from station
Ny-Ålesund (2017-04 et seq). Alfred Wegener Institute - Research Unit Potsdam,
PANGAEA, https://doi.org/10.1594/PANGAEA.914973

Parameter(s):
- DATE/TIME (Date/Time)
- LATITUDE (Latitude)  * METHOD/DEVICE: GPS receiver mounted on radiosonde RS41
- LONGITUDE (Longitude)  * METHOD/DEVICE: GPS receiver mounted on radiosonde
  RS41
- ALTITUDE [m] (Altitude)  * METHOD/DEVICE: integrated from pressure
  and temperature * COMMENT: Geopotential height above sea level
- Height, geometric [m] (h geom) *METHOD/DEVICE: GPS receiver mounted on
  radiosonde RS41 * COMMENT: with respect to WGS84 reference ellipsoid
- Elapsed time [s] (ETIM) *
- Pressure, at given altitude [hPa] (PPPP) *  * METHOD/DEVICE: Radiosonde,
  Vaisala, RS41 * COMMENT: combined uncertainty 1.0 hPa (>100 hPa), 0.6 hPa
  (100-3 hPa)
- Temperature, air [°C] (TTT) *  * METHOD/DEVICE: Radiosonde, Vaisala, RS41 *
  COMMENT: combined uncertainty 0.2-0.4 K
- Humidity, relative [%] (RH) *  * METHOD/DEVICE: Radiosonde, Vaisala, RS41 *
  COMMENT: combined uncertainty 3-4 %
- Wind direction [deg] (dd) *  * METHOD/DEVICE: Calculated from GPS
- Wind speed [m/s] (ff) *  * METHOD/DEVICE: Calculated from GPS
"""

import datetime
import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv


PATH = os.path.join(os.environ["PATH_SEC"], "data/radiosondes/ny_alesund/")


def main():
    """
    Conversion of PANGAEA radiosonde data.
    """

    #to_netcdf()

    merge(
        t0="2017-04-01 00:00:00",
        t1="2023-12-31 23:59:59",
    )


def to_netcdf():
    """
    Creates NetCDF file for each sounding
    """

    files = sorted(
        glob(
            os.path.join(
                PATH,
                "monthly/NYA_radiosonde_*.tab",
            )
        )
    )

    for i_file, file in enumerate(files):
        print(f"{i_file + 1}/{len(files)}")

        # read header and extract global attributes
        i = 0
        bag = []
        descr = {"Citation": "", "Project(s)": ""}
        with open(file, "r") as f:
            for i, line in enumerate(f):
                line_txt = line.strip()

                bag.extend([line_txt])

                i += 1
                if line_txt == "*/":
                    break

        global_attrs = "\n".join(bag)

        # read data of file
        data = pd.read_csv(file, skiprows=i, sep="\t")
        data["Date/Time"] = pd.to_datetime(
            data["Date/Time"], format="%Y-%m-%dT%H:%M:%S"
        )

        # rename columns
        data = data.rename(
            columns={
                "Date/Time": "time",
                "Latitude": "lat",
                "Longitude": "lon",
                "Altitude [m]": "geopot",
                "h geom [m]": "alt",
                "ETIM [s]": "etim",
                "PPPP [hPa]": "press",
                "TTT [°C]": "temp",
                "RH [%]": "rh",
                "dd [deg]": "wdir",
                "ff [m/s]": "wspeed",
            }
        )

        data = data.set_index("time")

        # convert to xarray and add attributes
        ds_rsd = data.to_xarray()

        # remove unrealistic values
        ds_rsd["rh"] = xr.where(
            ds_rsd["rh"] < 150, ds_rsd["rh"], np.nan
        )

        # set to realistic range
        ds_rsd["rh"] = xr.where(
            ds_rsd["rh"] < 100, ds_rsd["rh"], 100
        )

        ds_rsd["rh"] = xr.where(
            ds_rsd["rh"] > 0, ds_rsd["rh"], 0
        )

        # global attributes
        ds_rsd.attrs = dict(
            description=global_attrs,
            history="Downloaded from PANGAEA and transformed "
            "to netcdf by Nils Risse",
            created=str(np.datetime64("now")),
        )

        # set local attributes
        ds_rsd["time"].attrs = dict(standard_name="time")
        ds_rsd["time"].encoding = dict(units="seconds since 1970-01-01")
        ds_rsd["etim"].attrs = dict(
            standard_name="elapsed_time",
            long_name="elapsed time since launch",
            units="s",
        )

        ds_rsd["press"].attrs = dict(standard_name="air_pressure", units="hPa")
        ds_rsd["temp"].attrs = dict(
            standard_name="air_temperature", units="°C"
        )
        ds_rsd["rh"].attrs = dict(standard_name="relative_humidity", units="%")
        ds_rsd["wdir"].attrs = dict(
            standard_name="wind_from_direction", units="degree"
        )
        ds_rsd["wspeed"].attrs = dict(
            standard_name="wind_speed", units="m s-1"
        )

        ds_rsd["geopot"].attrs = dict(
            standard_name="geopotential_height", units="m"
        )
        ds_rsd["alt"].attrs = dict(
            standard_name="altitude",
            long_name="GPS height above WGS84 reference" " ellipsoid",
            units="m",
        )

        ds_rsd["lon"].attrs = dict(
            standard_name="longitude", units="degrees_east"
        )
        ds_rsd["lat"].attrs = dict(
            standard_name="latitude", units="degrees_north"
        )

        # identify individual soundings
        delta_etim = ds_rsd.etim.values[1:] - ds_rsd.etim[:-1]
        # index, where new sounding starts (between soundings only)
        ix = np.argwhere(delta_etim.values < 0).flatten()
        # add start of first sounding (-1 will be gone later, but this is end)
        ix = np.append(np.array([-1]), ix)
        # add end of last sounding
        ix = np.append(ix, np.array([len(delta_etim)]))

        # save individual soundings to file
        total_length = len(ds_rsd.time)
        piece_length = 0
        for i in range(len(ix) - 1):
            i_start, i_end = [ix[i], ix[i + 1]]

            ds = ds_rsd.isel(time=slice(i_start + 1, i_end + 1))

            # roughly the length of one sounding in hours
            assert (i_end - i_start) / 3600 < 4

            date_launch = datetime.datetime.fromtimestamp(
                ds.time[0].values.astype("int") * 1e-9
            )
            date_str = date_launch.strftime("%Y%m%d%H%M")
            ds.to_netcdf(
                os.path.join(
                    PATH,
                    f"radiosonde_ny_alesund_{date_str}.nc",
                )
            )

            piece_length += len(ds.time)

        assert total_length == piece_length, "Some data is lost!"


def merge(t0, t1):
    """
    Merge all radiosondes from Ny-Alesund onto a common grid.
    """

    # get all radiosondes
    dates = pd.date_range(t0, t1, freq="1D")

    files = []
    for date in dates:
        files.extend(
            glob(
                os.path.join(
                    PATH,
                    f"radiosonde_ny_alesund_{date.strftime('%Y%m%d')}*.nc",
                )
            )
        )

    # read all sondes on common grid
    ds_dsd = xr.open_mfdataset(
        files, preprocess=prep, concat_dim="i_sonde", combine="nested"
    )

    # remove sondes with too many nans
    ds_dsd = ds_dsd.sel(i_sonde=ds_dsd.nans < 30)

    # save to file
    ds_dsd.to_netcdf(
        os.path.join(
            PATH,
            "merge/radiosonde_ny_alesund_merge.nc",
        )
    )


def prep(ds):
    """
    Preprocessing for open_mfdataset that adds dimension along which to
    concatenate

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of a single sounding
    """

    n_lev = 200
    z_even = np.linspace(
        10, 33000, n_lev, endpoint=True
    )  # 0.1 to 33,000 meters
    factor = np.linspace(
        0.05, 1, n_lev, endpoint=True
    )  # factor to change dz: lower at low altitude, higher at high altitude
    z_lev = z_even * factor

    ds_atm_raw = xr.Dataset()
    ds_atm_raw.coords["z_lev"] = ds.alt.values
    ds_atm_raw["temp"] = (("z_lev"), ds.temp.values)
    ds_atm_raw["rh"] = (("z_lev"), ds.rh.values)
    ds_atm_raw["press"] = (("z_lev"), ds.press.values)

    # remove duplicate indices and nans
    _, index = np.unique(ds_atm_raw["z_lev"], return_index=True)
    ds_atm_raw = ds_atm_raw.isel({"z_lev": index})
    ds_atm_raw = ds_atm_raw.dropna(dim="z_lev")
    ds_atm_raw = ds_atm_raw.isel(z_lev=~np.isnan(ds_atm_raw["z_lev"]))

    # sort index
    ds_atm_raw = ds_atm_raw.sortby("z_lev")

    # interpolate on common grid
    ds_atm = ds_atm_raw.interp(coords={"z_lev": z_lev}, method="linear")

    # print number of nan's found in the different variables
    n_nan = np.array([])
    for var in list(ds_atm):
        n_nan = np.append(n_nan, np.sum(np.isnan(ds_atm[var])).values.item())

    n_nan = int(np.max(n_nan))

    # extrapolate
    # for pressure fit logarithmic curve
    if np.any(np.isnan(ds_atm["press"])):
        p_avail = ~np.isnan(ds_atm["press"])
        a, b, c, d, e = np.polyfit(
            ds_atm["z_lev"].isel({"z_lev": p_avail}),
            ds_atm["press"].isel({"z_lev": p_avail}),
            4,
        )

        ds_atm["press"][~p_avail] = (
            a * ds_atm["z_lev"][~p_avail] ** 4
            + b * ds_atm["z_lev"][~p_avail] ** 3
            + c * ds_atm["z_lev"][~p_avail] ** 2
            + d * ds_atm["z_lev"][~p_avail] ** 1
            + e
        )

    ds_atm["press"][ds_atm["press"] < 0.4] = 0.4  # min in PAMTRA is 0.1hPa

    # others extrapolate linearly
    ds_atm = ds_atm.bfill(dim="z_lev")
    ds_atm = ds_atm.ffill(dim="z_lev")

    # add dimension that counts all overpasses
    ds_atm = ds_atm.expand_dims(dim="i_sonde", axis=0)
    ds_atm["time"] = (("i_sonde"), [ds.time.isel(time=0).values])

    # add number of nan (highes)
    ds_atm["nans"] = (("i_sonde"), [n_nan])

    return ds_atm


if __name__ == "__main__":
    main()
