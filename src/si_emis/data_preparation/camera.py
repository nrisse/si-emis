"""
Plot fish-eye data that is uploaded on PANGAEA to png image with metadata

Caution:
    viewing zenith angle is not always scaled correctly in the dataset.
    This has to be filtered and corrected (usually 1e-2, sometimes 1e-6)
    Also, the values are sometimes defective, e.g., in
    NIKON_radiance_P5_20190321_14UTC.nc

    therefore use a constant vza from one file for each campaign that seems
    to work good (they are anyway constant at least for each camera/campaign)

    sometimes the date in the fish-eye images is wrong, therefore always
    use the date of the ac3airborne flight dict
"""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lizard.ac3airlib import day_of_flight

from si_emis.readers.fish_eye import files_fish_eye_nc, read_fish_eye_nc

CIRCLE_RADIUS = 100  # circle radius displayed in center of plot in m
FLIGHT_IDS = [
    "AFLUX_P5_RF08",
    "AFLUX_P5_RF14",
    "AFLUX_P5_RF15",
]


def main():
    """
    Plot camera images from NetCDF files available on PANGAEA
    """

    for flight_id in FLIGHT_IDS:
        print(flight_id)

        vza = get_vza(flight_id)

        # get all files
        files = files_fish_eye_nc(flight_id)

        for file in files:
            print(file)

            ds_eye, radiance_scaling = read_fish_eye_nc(file)

            # plot fish-eye image
            for t in ds_eye.dim_t.values:
                # plot rgb with outer angle of 75 deg
                """
                plot_full(
                        flight_id=flight_id,
                        ds_eye=ds_eye,
                        t=t,
                        vza=vza,
                        outer_angle=75,
                        radiance_scaling=radiance_scaling,
                        angle_offset=None,
                        add_scale=False,
                        add_contours=True,
                        add_annotations=True,
                        zoom='fixed',
                        prefix='fish_eye')
                """

                # plot rgb with outer angle of 60 deg and no annotation
                """
                plot_full(
                        flight_id=flight_id, 
                        ds_eye=ds_eye, 
                        t=t, 
                        vza=vza, 
                        outer_angle=60, 
                        radiance_scaling=radiance_scaling, 
                        angle_offset=None, 
                        add_scale=True, 
                        add_contours=False, 
                        add_annotations=False, 
                        zoom='fixed', 
                        prefix='no_annot_60_fish_eye')
                """

                # plot with outer angle altitude-dependent
                plot_full(
                    flight_id=flight_id,
                    ds_eye=ds_eye,
                    t=t,
                    vza=vza,
                    outer_angle=None,
                    radiance_scaling=radiance_scaling,
                    max_angle=30,
                    add_scale=True,
                    add_contours=False,
                    add_annotations=False,
                    zoom="flexible",
                    prefix="no_annot_flexible_fish_eye",
                )

                plt.close("all")


def plot_full(
    flight_id,
    ds_eye,
    t,
    vza,
    outer_angle,
    radiance_scaling=None,
    max_angle=30,
    add_scale=True,
    add_contours=False,
    add_annotations=False,
    zoom="fixed",
    prefix="fish_eye",
):
    """
    Plots fish-eye lens RGB image. Either a fixed angle is plotted or a zoomed
    in image depending on the flight altitude.

    fixed zoom: outer angle is fixed (e.g. 60 deg)
    flexible zoom: outer angle depends on flight altitude (e.g. zenith angle
    of 100 m diameter plus 3 deg)

    Parameters
    ----------
    flight_id : str
        ac3airborne flight id
    ds_eye : xarray.Dataset
        fish eye dataset
    t : int
        time step to be plotted (from xarray dataset dimension)
    vza : numpy.ndarray
        viewing zenith angle 2D array
    outer_angle : int
        outer angle to clip image
    radiance_scaling : _type_, optional
        _description_, by default None
    max_angle : int, optional
        _description_, by default 30
    add_scale : bool, optional
        _description_, by default True
    add_contours : bool, optional
        _description_, by default False
    add_annotations : bool, optional
        _description_, by default False
    zoom : str, optional
        _description_, by default 'fixed'
    prefix : str, optional
        _description_, by default 'fish_eye'
    """

    mission, platform, name = flight_id.split("_")

    # measurement time
    time = datetime.datetime.utcfromtimestamp(
        (
            ds_eye.time.sel(dim_t=t).values
            - np.datetime64("1970-01-01 00:00:00")
        )
        / np.timedelta64(1, "s")
    )
    time_str1 = time.strftime("%H%M%S")
    time_str2 = time.strftime("%H:%M:%S")

    date_str = day_of_flight(flight_id).strftime("%Y%m%d")

    # rgb image
    rgb, max_radiance = prepare_rgb(ds_eye, t)

    # clip image
    if zoom == "fixed":
        # limit to outer angle to have a round edge
        within_angle = vza <= outer_angle
        rgb[~within_angle, :] = 255

    elif zoom == "flexible":
        theta = zenith_angle(
            height=ds_eye.altitude.sel(dim_t=t).values.item(), diameter=1000
        )
        outer_angle = np.min([theta, max_angle])

        # limit to outer angle to have a round edge
        within_angle = vza <= outer_angle
        rgb[~within_angle, :] = 255

    # image boundary
    center = int(vza.shape[0] / 2)
    radius = int((vza.shape[0] - (~within_angle).all(axis=0).sum()) / 2)

    fig, ax = plt.subplots(
        1, 1, figsize=(5, 5), constrained_layout=True, frameon=False
    )
    ax.axis("off")

    rgba = np.concatenate(
        [rgb, 255 * within_angle.astype("int")[:, :, np.newaxis]], axis=-1
    )

    im = ax.imshow(rgba, interpolation="nearest")

    # image origin is upper left corner
    ax.set_xlim(center - radius, center + radius)
    ax.set_ylim(center + radius, center - radius)

    if add_scale:
        # plot circle with 100 m diameter
        theta = zenith_angle(
            height=ds_eye.altitude.sel(dim_t=t).values.item(),
            diameter=CIRCLE_RADIUS,
        )

        ax.contour(
            np.arange(vza.shape[0]),
            np.arange(vza.shape[1]),
            vza,
            levels=np.array([theta]),
            colors="coral",
            linestyles="-",
            linewidths=5,
        )

    if add_contours:
        # show lines for angles
        ct_vza = ax.contour(
            np.arange(vza.shape[0]),
            np.arange(vza.shape[1]),
            vza,
            levels=np.array([10, 40, 70]),
            colors="#B50066",
            linestyles="-",
            linewidths=0.75,
        )

        # add labels to contours (azimuth not needed)
        ax.clabel(
            ct_vza,
            fmt="%1.0f °",
            manual=[(center, 490), (center, 640), (center, 800)],
        )

    if add_annotations:
        lat = ds_eye.latitude.sel(dim_t=t).values.item()
        lon = ds_eye.longitude.sel(dim_t=t).values.item()
        height = ds_eye.altitude.sel(dim_t=t).values.item()

        # add flight altitude, time, position as text
        txt = "{} °N\n{} °E\n{} m".format(
            np.round(lat, 3), np.round(lon, 3), int(np.round(height, 0))
        )
        ax.annotate(
            txt,
            xy=(center - radius, center - radius),
            xycoords="data",
            ha="left",
            va="top",
        )

        ax.annotate(
            r"$I_{\lambda, 255}$=%1.3f\nWm$^{-2}$nm$^{-1}$sr"
            r"$^{-1}$" % (max_radiance * radiance_scaling),
            xy=(center + radius, center - radius),
            xycoords="data",
            ha="right",
            va="top",
        )

        time_txt = f"{mission} {name}, {date_str} {time_str2} UTC"
        ax.set_title(time_txt)

    plt.savefig(
        os.path.join(
            os.environ["PATH_DAT"],
            f"obs/campaigns/{mission.lower()}/p5",
            f"radiance_fields_png/{name}",
            f"{prefix}_{mission}_{name}_{date_str}_{time_str1}.png",
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )


def prepare_rgb(ds_eye, t):
    """
    Prepares RGB camera image.

    Parameters
    ----------
    ds_eye : xarray.Dataset
        fish-eye lense camera dataset
    t : int
        time integer of dataset

    Returns
    -------
    numpy.ndarray
        RGB image array
    """

    rgb = np.stack(
        [
            ds_eye[c].sel(dim_t=t)
            for c in ["radiance_red", "radiance_green", "radiance_blue"]
        ]
    )
    max_radiance = np.percentile(rgb, 95)
    rgb = scaling(
        x=rgb.astype("float64"), a=0, b=255, x_min=0, x_max=max_radiance
    )
    rgb = np.round(rgb, 0).astype("int")
    rgb = rgb.transpose(1, 2, 0)

    return rgb, max_radiance


def zenith_angle(height, diameter):
    """
    Calculates the zenith angle of the circle outline with a given diameter
    and observed from a given height.

    Parameters
    ----------
    height : float64
        aircraft altitude in meters
    diameter : float64
        circle diameter in meters

    Returns
    -------
    float64
        zenith angle in degrees
    """

    theta = np.arctan(diameter / 2 / height)
    theta = np.rad2deg(theta)

    return theta


def scaling(x, a, b, x_min=None, x_max=None):
    """
    Feature scaling

    Scale value x in range from a to b

    x_min: minimum value of x, that may not be represented of subset x
    x_max: maximum value of x, that may not be represented of subset x
    """

    if x_min is None:
        x_min = np.min(x)

    if x_max is None:
        x_max = np.max(x)

    x_scaled = a + (x - x_min) * (b - a) / (x_max - x_min)

    return x_scaled


def roundup(x):
    return int(np.ceil(x / 100.0)) * 100


def get_vza(flight_id):
    """
    Read viewing zenith angle for one mission

    Comment: only checked for AFLUX
    """

    mission, platform, name = flight_id.split("_")

    path = os.path.join(
        os.environ["PATH_DAT"],
        f"obs/campaigns/{mission.lower()}/p5/radiance_fields/",
    )

    if mission == "ACLOUD":
        file = path + "/Canon_radiance_P5_170625_14UTC_Flight_23.nc"
        scale = 1e-2

    elif mission == "AFLUX":
        file = path + "NIKON_radiance_P5_20190321_12UTC.nc"
        scale = 1e-6

    elif mission == "MOSAiC-ACA":
        file = path + "NIKON_radiance_P5_20200911a_10UTC.nc"
        scale = 1e-6

    vza = xr.open_dataset(file).vza.values
    vza = vza * scale

    return vza


if __name__ == "__main__":
    main()
