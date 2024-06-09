"""
Emissivity case study during AFLUX RF08.

Sentinel RGB
------------
The Sentinel-2 natural color image was created from channels 4, 3, and 2. The
gdal command is:
gdal_merge.py -o rgb.tiff -separate B04.jp2 B03.jp2 B02.jp2
"""

import os
import string

import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import yaml
from cartopy import crs as ccrs
from dotenv import load_dotenv
from osgeo import gdal, osr
from scipy import ndimage

from lizard.ac3airlib import meta
from lizard.readers.band_pass import read_band_pass_combination
from lizard.readers.worldview import read_rgb
from lizard.writers.figure_to_file import write_figure
from si_emis.data_preparation.airsat import airborne_filter
from si_emis.figures.figure_05_kmeans import apply_kmeans
from si_emis.readers.emissivity import read_emissivity_aircraft
from si_emis.readers.fish_eye import read_fish_eye
from si_emis.style import mirac_cols, mirac_pol_text

load_dotenv()

FLIGHT_ID = "AFLUX_P5_RF08"
CASE_NAME = "aflux_rf08_case"
SURF_REFL = "L"
DIST_MAX = 11000  # this should be the distance between start and end point
CHANNELS = [1, 7, 8, 9]
CHANNEL_GROUPS = {1: [1], 7: [2, 3, 4, 5, 6, 7], 8: [8], 9: [9]}
Y_LIMITS_E = {1: (0.6, 1), 7: (0.6, 1), 8: (0.6, 1), 9: (0.6, 1)}
Y_LIMITS_TB = {1: (160, 250), 7: (190, 260), 8: (190, 250), 9: (200, 250)}


def main():
    """
    Overview plot of a case study with Tb, emissivity, Ts, satellite, and
    camera images.
    """

    # read data
    ds = prepare_data()

    # read mirac band pass
    ds_bp = read_band_pass_combination()

    # add label to dataset
    ds["label"] = ds_bp.label

    # get variability of emissivity and tb
    statistics_print(ds)

    # read satellite image
    file_sat = "sentinel/AFLUX_RF08_20190331/rgb_shifted_crop.tiff"
    ds_landsat, ds_landsat_data, projection, ds_landsat_gt = open_imagery(
        os.path.join(
            os.environ["PATH_SEC"], "data/sea_ice_emissivity", file_sat
        )
    )

    # times of pre-created fish-eye images
    camera_times = [
        np.datetime64("2019-03-31 11:42:16"),
        np.datetime64("2019-03-31 11:41:32"),
        np.datetime64("2019-03-31 11:40:20"),
        np.datetime64("2019-03-31 11:40:00"),
    ]
    images = read_camera_images(camera_times)
    image_labels = ["I", "II", "III", "IV"]
    image_dct = {
        image_labels[i]: (camera_times[i], images[i])
        for i in range(len(camera_times))
    }

    # get location of camera images for plot
    special_location = []
    for time in camera_times:
        special_location.append(
            [
                ds["lon"].sel(time=time, channel=2).item(),
                ds["lat"].sel(time=time, channel=2).item(),
            ]
        )

    overview_plot(
        ds,
        image_dct,
        ds_landsat,
        ds_landsat_data,
        ds_landsat_gt,
        projection,
        special_location,
    )


def open_imagery(file):
    """
    Script to read landsat image

    input: full path of landsat image
    """

    ds = gdal.Open(file)

    # read and convert to uint8
    ds_data = ds.ReadAsArray()
    ds_data = ds_data.astype(np.float32)

    ds_gt = ds.GetGeoTransform()
    ds_proj = ds.GetProjection()

    # get projection
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(ds_proj)

    # convert the WKT projection information to a cartopy projection.
    projcs = inproj.GetAuthorityCode("PROJCS")
    projection = ccrs.epsg(projcs)

    # transpose as required for matplotlib
    ds_data = ds_data.transpose((1, 2, 0))

    ds_data[:, :, 0] = normalize(ds_data[:, :, 0])
    ds_data[:, :, 1] = normalize(ds_data[:, :, 1])
    ds_data[:, :, 2] = normalize(ds_data[:, :, 2])

    return ds, ds_data, projection, ds_gt


def normalize(band):
    """
    Normalizes band to 0-1 range.
    """

    band_min, band_max = (band.min(), band.max())

    return (band - band_min) / (band_max - band_min)


def plot_imagery(ax, ds, ds_data, ds_gt):
    """
    Plot landsat image on an empty axis
    """

    extent = (
        ds_gt[0],
        ds_gt[0] + ds.RasterXSize * ds_gt[1],
        ds_gt[3] + ds.RasterYSize * ds_gt[5],
        ds_gt[3],
    )

    img = ax.imshow(ds_data, extent=extent, origin="upper", vmin=0, vmax=0.16)

    return img


def imagery_on_map(
    ax,
    ds,
    ds_data,
    ds_gt,
    projection,
    ds_rad,
    case_name,
    special_location=[],
    width_height_ratio=0.4,
):
    """
    Plot landsat image on map and return as image

    Possible flights: ACLOUD RF23, AFLUX RF15

    Important: if ds_rad changes, map extent might also change!

    Parameters
    ----------
    ax: axis on which the rotated map will be plotted
    ds: satellite image
    ds_data: satellite image data
    ds_gt: satellite image geotransform
    projection: satellite image projection
    ds_rad: airborne observations (should contain lon, lat, and ts for every
      channel)
    case_name: case study name
    special_location: coordinates of special location to be indicated on the
      map. For example camera images. Provide coordinates in a list for
      each location to be indicated. [[lon0, lat0], [lon1, lat1], ...]
    """

    if case_name == "aflux_rf08_case":
        x0, x1 = [240, 2980]
        y0, y1 = [2177, 830]
        rot = 0.2

    else:
        x0, x1, y0, y1, rot = [None, None, None, None, None]

    y_cen = int((y0 + y1) / 2)
    x_width = x1 - x0
    y0, y1 = [
        y_cen + int(x_width * width_height_ratio / 2),
        y_cen - int(x_width * width_height_ratio / 2),
    ]

    fig0, ax0 = plt.subplots(
        1, 1, figsize=(10, 10), subplot_kw=dict(projection=projection), dpi=300
    )

    # plot landsat image and flight track
    plot_imagery(ax0, ds, ds_data, ds_gt)
    im = ax0.scatter(
        ds_rad.lon.sel(channel=8),
        ds_rad.lat.sel(channel=8),
        c=ds_rad.ts.sel(channel=8),
        s=30,
        cmap=cmc.batlow,
        transform=ccrs.PlateCarree(),
        vmin=248,
        vmax=255,
    )
    ax0.axis("off")

    # indicate special locations
    for sloc in special_location:
        ax0.scatter(
            sloc[0],
            sloc[1],
            edgecolor="gray",
            facecolor="None",
            s=6000,
            linewidth=3,
            transform=ccrs.PlateCarree(),
        )

    # get image and rotate it
    fig0.canvas.draw()
    image = np.frombuffer(fig0.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig0.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig0)

    image_rotated = ndimage.rotate(image, rot)

    # plot on new axis
    ax.imshow(image_rotated[y1:y0, x0:x1, :], cmap="Greys")
    ax.axis("off")

    return ax, im


def helper_find_crop(image_rotated, outfile):
    """This function plots the image to find the crop values"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_rotated)
    ax.set_xticks(np.arange(0, 3000, 10))
    ax.set_yticks(np.arange(0, 3000, 10))
    ax.set_xticklabels(np.arange(0, 3000, 10), fontsize=4, rotation=90)
    ax.set_yticklabels(np.arange(0, 3000, 10), fontsize=4)
    ax.grid()
    plt.savefig(outfile, dpi=1200)


def prepare_data():
    """
    Prepare emissivity dataset for overview plot.

    The following steps are conducted:
    - read and filter emissivity
    - calculate distance along linear flight track
    - slice selected time window

    Returns
    -------
    ds: emissivity dataset reduced to the distance that will be used for
      plotting.
    """

    ds = read_emissivity_aircraft(flight_id=FLIGHT_ID, without_offsets=False)
    ds = airborne_filter(ds, drop_times=True, dtb_keep_tb=True)

    # calculate distance between two points along flight track
    ds = calculate_distance(ds, case_name=CASE_NAME, epsg=32631)

    # reduce to segment for case study
    segment = meta(FLIGHT_ID)["segments"][9]["parts"][-1]
    ds = ds.sel(time=slice(segment["start"], segment["end"]))

    # remove observations that lie outside of range
    ds["e"] = ds["e"].where(ds.dist < DIST_MAX)
    ds["e"] = ds["e"].where(ds.dist > 0)

    # use only lambertian emissivity
    ds = ds.sel(surf_refl=SURF_REFL).reset_coords(drop=True)

    # keep only times where at least one emissivity is available
    ds = ds.sel(time=ds.e.notnull().any(["channel", "i_offset"]))

    print(ds.time[0].values)
    print(ds.time[-1].values)

    print(ds.lon.min().item())
    print(ds.lon.max().item())
    print(ds.lat.min().item())
    print(ds.lat.max().item())

    return ds


def statistics(ds):
    """
    This function shows some quicklooks
    """

    ds["km_label"] = apply_kmeans().km_label

    # time series of emissivity and cluster number
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all")

    for channel in CHANNELS:
        ax1.scatter(
            ds.dist.sel(channel=channel) * 1e-3,
            ds.e.sel(channel=channel),
            color=mirac_cols[channel],
        )

    ax2.scatter(
        ds.dist.sel(channel=1) * 1e-3, ds.km_label, color=mirac_cols[1]
    )

    # time series of tbs
    fig, ax = plt.subplots(1, 1)

    for channel in CHANNELS:
        ax.scatter(
            ds.dist.sel(channel=channel) * 1e-3,
            ds.tb.sel(channel=channel),
            color=mirac_cols[channel],
        )


def statistics_print(ds):
    """
    This function calculates simple statistics along this transect to idenfity
    the sea ice temperature and tbs.
    """

    # identify, which variables causes highest uncertainty
    no_offset = (ds.offset == 0).all("variable")
    ds["de"] = ds.e - ds.e.sel(i_offset=no_offset).squeeze()
    bin_edges = np.arange(-0.5, 0.5 + len(ds.offset))
    x = np.abs(ds["de"]).idxmax("i_offset").values
    hist = np.apply_along_axis(
        lambda a: np.histogram(a, bins=bin_edges)[0], 0, x
    )
    ix_offset_max_error = hist.argmax(axis=0)
    ix_offset_max_error = ds.offset.isel(i_offset=ix_offset_max_error)
    print(ix_offset_max_error)

    # what is the relative uncertainty for each source
    da_rel_bias = (ds["de"] / ds["e"] * 100).mean("time")
    da_rel_bias = da_rel_bias.sel(i_offset=~no_offset)

    # add relative bias due to tb
    da_rel_bias_tb = (
        ds["tb_unc"] * 1 / ds["dtb"].sel(i_offset=no_offset) / ds["e"] * 100
    ).mean("time")
    da_rel_bias_tb["i_offset"] = np.array([7])

    da_rel_bias = xr.concat([da_rel_bias, da_rel_bias_tb], dim="i_offset")

    # change order to have similar variables next to each other
    da_rel_bias = da_rel_bias.sel(i_offset=[6, 0, 5, 1, 4, 2, 7])

    # do not show channels where emissivity is nan
    da_rel_bias = da_rel_bias.sel(channel=~da_rel_bias.isnull().all("i_offset"))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    cmap = cmc.berlin
    norm = mcolors.BoundaryNorm(np.arange(-15, 16, 2.5), cmap.N)
    im = ax.pcolormesh(
        np.arange(0, len(da_rel_bias.channel)),
        np.arange(0, len(da_rel_bias.i_offset)),
        da_rel_bias.T,
        cmap=cmap,
        norm=norm,
        shading="nearest",
    )
    fig.colorbar(
        im,
        label="Relative emissivity change [%]",
        ticks=np.arange(-15, 15.1, 5),
        extend="both",
    )

    ax.set_xticks(np.arange(0, len(da_rel_bias.channel)))
    ax.set_yticks(np.arange(0, len(da_rel_bias.i_offset)))

    ax.set_xticklabels(list(ds.label.sel(channel=da_rel_bias.channel).values), rotation=90)

    # y ticks are the uncertainties and the related offsets applied
    rename = {"groundtemp": "$T_s$", "temp": "$T$", "relhum": "$RH$"}
    unit = {"groundtemp": "K", "temp": "K", "relhum": "%"}
    labels = []
    for i in da_rel_bias.i_offset.values:
        if i in ds.i_offset.values:
            for v in ds.variable.values:
                offset = ds.offset.sel(i_offset=i, variable=v).item()
                if offset != 0:
                    txt = f"{rename[v]}: {offset:+} {unit[v]}"

        else:
            txt = "$T_b$: 0.5 K (2.5 K at 89 GHz)"

        labels.append(txt)

    ax.set_yticklabels(labels)

    write_figure(fig, "relative_emissivity_change_case_rf08.png")

    # statistics of ts, tb, and emissivity
    for d0, d1 in [[0, 11e3], [0, 7e3], [7e3, 11e3]]:
        print(f"{d0*1e-3} km - {d1*1e-3} km")
        for v in ["ts", "tb", "e"]:
            for channel in CHANNELS:
                mean = (
                    ds[v]
                    .sel(
                        time=(
                            (ds.dist.sel(channel=channel) > d0)
                            & (ds.dist.sel(channel=channel) < d1)
                        ),
                        channel=channel,
                    )
                    .mean("time")
                    .item()
                )
                mean = round(mean, 2)

                minn = (
                    ds[v]
                    .sel(
                        time=(
                            (ds.dist.sel(channel=channel) > d0)
                            & (ds.dist.sel(channel=channel) < d1)
                        ),
                        channel=channel,
                    )
                    .min("time")
                    .item()
                )
                minn = round(minn, 2)

                maxx = (
                    ds[v]
                    .sel(
                        time=(
                            (ds.dist.sel(channel=channel) > d0)
                            & (ds.dist.sel(channel=channel) < d1)
                        ),
                        channel=channel,
                    )
                    .max("time")
                    .item()
                )
                maxx = round(maxx, 2)

                minmax = round(maxx - minn, 2)

                std = (
                    ds[v]
                    .sel(
                        time=(
                            (ds.dist.sel(channel=channel) > d0)
                            & (ds.dist.sel(channel=channel) < d1)
                        ),
                        channel=channel,
                    )
                    .std("time")
                    .item()
                )
                std = round(std, 2)

                print(
                    f"Channel {channel}: {mean} {std} {minn} {maxx} {minmax}"
                    f" ({v})"
                )


def overview_plot(
    ds,
    image_dct,
    ds_landsat,
    ds_landsat_data,
    ds_landsat_gt,
    projection,
    special_location,
):
    """
    Overview plot of observations during a short flight segment over sea ice.

    Parameters
    ----------
    ds: emissivity dataset reduced to the time within the specified segment
      and with distance variable for each channel to plot on x-axis.
    image_dct:
    ds_landsat:
    ds_landsat_data:
    ds_landsat_gt:
    projection:
    special_location:
    """

    nrow_top = 1
    ncol_top = 4
    nrow_bottom = len(CHANNELS) + 1
    ncol_bottom = 2

    fig = plt.figure(figsize=(5.5, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5])
    gs0 = gs[0].subgridspec(nrow_top, ncol_top)
    gs1 = gs[1].subgridspec(nrow_bottom, ncol_bottom)

    for i in range(ncol_top):
        fig.add_subplot(gs0[0, i])

    for i in range(nrow_bottom):
        for j in range(ncol_bottom):
            fig.add_subplot(gs1[i, j])

    ax_sat = fig.axes[4]  # satellite image
    ax_cam = fig.axes[:4]  # camera images
    ax_ts = fig.axes[5]  # surface temperature
    ax_tb = fig.axes[6::2]  # tb
    ax_e = fig.axes[7::2]  # emissivity

    # plot camera images
    for i, (name, img_prop) in enumerate(image_dct.items()):
        time, img = img_prop

        ax_cam[i].imshow(img)
        ax_cam[i].axis("off")

        # add letter for each image
        ax_cam[i].annotate(
            name,
            xy=(1, 1),
            xycoords="axes fraction",
            ha="center",
            va="top",
            color="gray",
        )

        # add vertical line when image was taken to other axes
        for ax in [ax_ts] + ax_tb + ax_e:
            x = ds.dist.sel(time=time, channel=2) * 1e-3
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.5)

        # add annotation above vertical line on ts axis
        x = ds.dist.sel(time=time, channel=2) * 1e-3
        ax_ts.annotate(
            name,
            xy=(x, 1),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            color="gray",
        )

        # add letters on satellite axis
        loc = ds.dist.sel(time=time, channel=2) / (DIST_MAX)
        ax_sat.annotate(
            name,
            xy=(loc, 0.7),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            color="gray",
        )

    # add label for orange circle being 100 m diameter
    ax_cam[0].annotate(
        "100 m",
        xy=(0.5, 0.7),
        xycoords="axes fraction",
        ha="center",
        va="center",
        color="coral",
    )

    # surface temperature
    ax_ts.scatter(
        ds.dist.sel(channel=8) * 1e-3, ds.ts.sel(channel=8), color="k", s=4
    )
    ax_ts.set_xlim(0, DIST_MAX * 1e-3)
    ax_ts.set_ylim(248, 255)

    # emissivity of the other 183 GHz channels (this is here so that they are
    # shown in the background behind the wing 183 GHz channel)
    for channel in range(2, 7):
        e = ds.e.sel(channel=channel)
        e_l = ds.e.sel(channel=channel) - ds.e_unc.sel(channel=channel)
        e_u = ds.e.sel(channel=channel) + ds.e_unc.sel(channel=channel)
        ax_e[1].scatter(
            ds.dist.sel(channel=channel) * 1e-3,
            e,
            color=mirac_cols[channel],
            s=4,
            zorder=1,
        )

        ax_e[1].fill_between(
            x=ds.dist.sel(channel=channel) * 1e-3,
            y1=e_l,
            y2=e_u,
            color=mirac_cols[channel],
            alpha=0.4,
            lw=0,
            label="uncertainty",
            zorder=0,
        )

    # tb and emissivity
    for i, channel in enumerate(CHANNELS):
        # plot tb
        for channel_tb in CHANNEL_GROUPS[channel]:
            ax_tb[i].scatter(
                ds.dist.sel(channel=channel_tb) * 1e-3,
                ds.tb.sel(channel=channel_tb),
                color=mirac_cols[channel_tb],
                s=4,
                label=ds["label"].sel(channel=channel_tb).item()
                + f" ({mirac_pol_text[channel_tb]})",
            )

        # plot emissivity
        e = ds.e.sel(channel=channel)
        e_l = ds.e.sel(channel=channel) - ds.e_unc.sel(channel=channel)
        e_u = ds.e.sel(channel=channel) + ds.e_unc.sel(channel=channel)
        ax_e[i].scatter(
            ds.dist.sel(channel=channel) * 1e-3,
            e,
            color=mirac_cols[channel],
            s=4,
            zorder=1,
        )

        ax_e[i].fill_between(
            x=ds.dist.sel(channel=channel) * 1e-3,
            y1=e_l,
            y2=e_u,
            color=mirac_cols[channel],
            alpha=0.4,
            lw=0,
            label="uncertainty",
            zorder=0,
        )

        # ticks of emissivity axis
        ax_e[i].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax_e[i].yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        # ticks of tb axis
        if channel == 1:
            ax_tb[i].yaxis.set_major_locator(mticker.MultipleLocator(40))
        else:
            ax_tb[i].yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax_tb[i].yaxis.set_minor_locator(mticker.MultipleLocator(10))

        # xticks
        ax_tb[i].xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax_tb[i].xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax_e[i].xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax_e[i].xaxis.set_major_locator(mticker.MultipleLocator(2))

        # axis limits
        ax_tb[i].set_ylim(Y_LIMITS_TB[channel])
        ax_e[i].set_ylim(Y_LIMITS_E[channel])

        ax_e[i].set_xlim(0, DIST_MAX * 1e-3)
        ax_tb[i].set_xlim(0, DIST_MAX * 1e-3)

    # remove xticks of upper axis
    for ax in [ax_ts] + ax_tb[:-1] + ax_e[:-1]:
        ax.set_xticklabels([])

    # axis labels
    ax_tb[-1].set_xlabel("Distance [km]")
    ax_e[-1].set_xlabel("Distance [km]")
    for ax in ax_tb:
        ax.set_ylabel("$T_b$ [K]")
    for ax in ax_e:
        ax.set_ylabel("$e$")
    ax_ts.set_ylabel("$T_s$ [K]")

    ax_ts.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax_ts.yaxis.set_major_locator(mticker.MultipleLocator(3))
    ax_ts.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax_ts.xaxis.set_major_locator(mticker.MultipleLocator(2))

    # annotate letters
    for i, ax in enumerate(fig.axes):
        if i < 4:
            ha = "center"
        else:
            ha = "left"

        txt = f"({string.ascii_lowercase[i]})"
        ax.annotate(txt, xy=(0, 1), xycoords="axes fraction", ha=ha, va="top")

    # add legend with channels on the right
    for i, ax in enumerate(ax_tb):
        # get legend items from tb subplot
        handles, labels = ax.get_legend_handles_labels()

        # add create legend next to emissivity subplot
        leg = ax_e[i].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            frameon=False,
            handlelength=0,
            handletextpad=0,
            labelspacing=0.1,
        )

        # remove legend markers
        for item in leg.legend_handles:
            item.set_visible(False)

        # set legend color
        for text in leg.get_texts():
            c = ds.where(
                ds.label == text.get_text().split("GHz")[0] + "GHz", drop=True
            ).channel.item()
            text.set_color(mirac_cols[c])

        # add legends as artists so that they won't be overwritten later
        ax_e[i].add_artist(leg)

        if i == 3:
            # add legend for uncertainty
            leg = ax_e[0].legend(
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(0.98, 1.2),
            )
            leg.set_in_layout(False)

    _, _, w, h = ax_ts.get_position().bounds

    ax_sat, im = imagery_on_map(
        ax_sat,
        ds_landsat,
        ds_landsat_data,
        ds_landsat_gt,
        projection,
        ds,
        CASE_NAME,
        special_location,
        width_height_ratio=h / w * 1.22,  # the number is put manually
    )

    # add colorbar on satellite image with surface temperature
    fig.colorbar(
        im,
        ax=ax_sat,
        extend="both",
        location="left",
        ticks=[248, 250, 252, 254],
        label="$T_s$ [K]",
        pad=-0.2,
        aspect=15,
    )

    # this extra argument was needed to avoid that legend gets cutoff
    write_figure(
        fig, "emissivity_case.png", bbox_extra_artists=[ax_e[1].get_legend()]
    )

    plt.close()


def read_camera_images(times):
    """
    Reads camera images for pre-defined time steps
    Returns

    To check available camera times:
    get_all_camera_times(FLIGHT_ID, t0=segment['start'], t1=segment['end'])

    Parameters
    ----------
    times: times of camera images
    """

    images = []
    for time in times:
        img, dt, img_time = read_fish_eye(
            time, FLIGHT_ID, prefix="no_annot_flexible_fish_eye"
        )

        # rotate image to have flight direction from left to right
        img = np.rot90(img, k=3)

        images.append(img)

    return images


def calculate_distance(ds, case_name, utm=None, epsg=None):
    """
    Calculate distance along flight track. Note that it is assumed that
    latitude and longitude coordinates are 2D, i.e., a function of time and
    channel.

    Parameters
    ----------
    ds: xarray.Dataset with latitude and longitude coordinates.
    case_name: case study name
    utm : str, optional
        UTM zone. The default is 32 (valid from 6 to 12 deg E).
    epsg: epsg code

    Returns
    -------
    xarray.Dataset with distance between start and end points of the case.
    """

    with open("./src/si_emis/data_preparation/coordinates.yaml") as f:
        coordinates = yaml.safe_load(f)[case_name]

    dist = distance(
        lat=ds.lat.values,
        lon=ds.lon.values,
        lat0=coordinates["p1"][1],
        lon0=coordinates["p1"][0],
        lat1=coordinates["p2"][1],
        lon1=coordinates["p2"][0],
        utm=utm,
        epsg=epsg,
    )

    ds["dist"] = (("time", "channel"), dist)

    return ds


def distance(lat, lon, lat0, lon0, lat1, lon1, utm=None, epsg=None):
    """
    Calculates the distance between the points lat/lon and lat0/lon0 projected
    along the straight line connecting the points lat0/lon0 and lat1/lon1.
    lat0/lon0 is set as origin with positive values towards lat1/lon0 and
    negative values in the opposite direction.

    A projected coordinate system has to be specified as EPSG code

    Parameters
    ----------
    lat : int or np.array
        Latitude of point for which distance will be calculated.
    lon : int or np.array
        Longitude of point for which distance will be calculated.
    lat0 : int
        Latitude origin point.
    lon0 : int
        Longitude origin point.
    lat1 : int
        Latitude end point.
    lon1 : int
        Longitude end point.
    utm : str, optional
        UTM zone. The default is 32 (valid from 6 to 12 deg E).

    Returns
    -------
    d_p1_p1 : int or np.array
        Distance in meters within the projected coordinate system.
    """

    assert epsg is None or utm is None

    # define coordinate systems
    if epsg is None:
        cs_proj = ccrs.UTM(zone=utm)

    else:
        cs_proj = ccrs.epsg(epsg)

    cs_geog = ccrs.PlateCarree()

    # start and end points of track
    p0 = np.array([[lon0], [lat0]])
    p1 = np.array([[lon1], [lat1]])

    # project all coordinates to (x, y, z)
    p0_p = cs_proj.transform_points(x=p0[0, :], y=p0[1, :], src_crs=cs_geog)
    p1_p = cs_proj.transform_points(x=p1[0, :], y=p1[1, :], src_crs=cs_geog)
    pn_p = cs_proj.transform_points(x=lon, y=lat, src_crs=cs_geog)

    # calculate unit vector along track
    e_d = (p1_p - p0_p) / np.linalg.norm(p1_p - p0_p, axis=1)

    # project vector from start point to target points along axis of track
    dist = np.dot((pn_p - p0_p), e_d.T)[..., 0]

    return dist


if __name__ == "__main__":
    main()
