"""
Read plots of fish-eye camera
"""

import datetime
import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from lizard.ac3airlib import day_of_flight
from matplotlib import image as mpimg

INSTRUMENT = {
    "ACLOUD": "Canon",
    "AFLUX": "NIKON",
    "MOSAiC-ACA": "NIKON",
}


def read_fish_eye_nc(file):
    """
    Reads NetCDF file of radiance files

    To see available files, run: files_fish_eye_nc(flight_id)

    Parameters
    ----------
    file: filename

    Returns
    -------
    ds_eye: xarray Dataset with radiance fields
    """

    ds_eye = xr.open_mfdataset(
        file, preprocess=convert_time, chunks={"dim_t": 2}
    )

    # get radiance scaling factor
    radiance_scaling = float(ds_eye.radiance_red.attrs["scaling_factor"])

    return ds_eye, radiance_scaling


def files_fish_eye_nc(flight_id):
    """
    Checks all available fish eye NetCDF files for a specific flight

    Parameters
    ----------
    flight_id: ac3airborne flight id

    Returns
    -------
    files: list of files
    """

    mission, platform, name = flight_id.split("_")
    date = day_of_flight(flight_id)
    date_str = date.strftime("%Y%m%d")

    files = sorted(
        glob(
            os.path.join(
                os.environ["PATH_DAT"],
                f"obs/campaigns/{mission.lower()}/p5/radiance_fields",
                f"{INSTRUMENT[mission]}_radiance_P5_{date_str}_*.nc",
            )
        )
    )

    return files


def convert_time(ds):
    """
    Xarray preprocess routine
    """

    # filter erroneous times
    ds = ds.sel(dim_t=ds.time <= 24)

    # get date from global attributes
    date = datetime.datetime.strptime(ds.attrs["Comment"], "Date %Y-%m-%d")
    time = np.array(
        [
            (date + datetime.timedelta(hours=h)).replace(microsecond=0)
            for h in ds.time.values.astype("float")
        ]
    )

    ds["time"] = ("dim_t", time)

    return ds


def read_fish_eye(time, flight_id, prefix="no_annot_60_fish_eye"):
    """
    Read fish eye image of current flight closest to the provided time.

    Example:
    read_fish_eye('2017-06-25 16:59:29', 'ACLOUD_P5_RF23')

    Parameters
    ----------
    time: time stamp
    flight_id: ac3airborne flight id

    Returns
    -------
    img: image as numpy array
    dt: time difference between camera phot and input time stamp
    img_time: time of the image
    """

    file, img_time, dt = find_closest(time, flight_id, prefix)

    img = mpimg.imread(file)

    return img, dt, img_time


def find_closest(time, flight_id, prefix="no_annot_60_fish_eye"):
    """
    Finds the closest fish eye image for a given time and flight_id.
    Reads only images without annotation.

    Example:
    read_fish_eye('2017-06-25 16:59:29', 'ACLOUD_P5_RF23')

    Parameters
    ----------
    time: time stamp
    flight_id: ac3airborne flight id
    prefix: filename prefix of image

    Returns
    -------
    img_file: file of the image
    img_time: time of the image
    dt: time difference between camera phot and input time stamp
    """

    mission, platform, name = flight_id.split("_")

    files = sorted(
        glob(
            os.path.join(
                os.environ["PATH_DAT"],
                f"obs/campaigns/{mission.lower()}/p5/radiance_fields_png/{name}",
                f"{prefix}_{mission}_{name}_*.png",
            )
        )
    )

    # find the nearest time index
    times = pd.to_datetime([x[-19:-4] for x in files], format="%Y%m%d_%H%M%S")
    ix = times.get_indexer([pd.Timestamp(time)], method="nearest")[0]
    img_time = times[ix]
    img_file = files[ix]

    # compute time difference
    dt = (img_time - time).total_seconds()

    return img_file, img_time, dt


def get_all_camera_times(flight_id, t0, t1):
    """
    Get all available camera times within time window

    Parameters
    ----------
    flight_id: flight_id
    t0: start time
    t1: end time

    Returns
    -------
    times: list of available times
    """

    files = files_fish_eye_nc(flight_id)

    for i, file in enumerate(files):
        print(i, file)

    i = int(input("Give index of file: "))

    ds_cam, radiance_scaling = read_fish_eye_nc(files[i])
    ds_cam = ds_cam.swap_dims({"dim_t": "time"})
    ds_cam = ds_cam.sel(time=slice(t0, t1))

    times = list(ds_cam.time.values)

    return times
