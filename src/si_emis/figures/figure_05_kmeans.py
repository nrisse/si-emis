"""
Performs a K-Means clustering in channel space using 89, 183, 243, and 340 GHz.

Note that the first cluster is labelled as 1 and not the default value of 0!
"""

import string

import cartopy.crs as ccrs
import cmcrameri as cmc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from lizard.writers.figure_to_file import write_figure
from matplotlib import cm
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from xhistogram.xarray import histogram

from si_emis.data_preparation.airsat import airborne_emissivity_merged
from si_emis.data_preparation.matching import Matching
from si_emis.readers.fish_eye import read_fish_eye
from si_emis.style import *

np.random.seed(15)

# dictionary with old class name as keys and new class name as values
RELABEL_DCT = {1: 1, 2: 3, 3: 4, 4: 2}

# slice of camera image for plot
X_DIM, Y_DIM = 1474, 1474  # image dimensions


def main():
    """
    Apply K-Means classification of emissivity spectra
    """

    # apply kmeans
    ds_ea = apply_kmeans()

    # statistics
    occurrence_statistics(ds_ea)

    # plot camera images
    # view_camera_random(ds_ea)
    # view_camera_nearest(ds_ea)

    # plot final kmeans spectra
    plot_kmeans_spectra(ds_ea)
    plot_kmeans_spectra_dist(ds_ea)

    # plot tb the same way
    plot_kmeans_spectra_violin_tb(ds_ea)

    # compute emissivity 25 and 75 percentiles for each cluster
    plot_quantiles(ds_ea, target_var="e")
    plot_quantiles(ds_ea, target_var="tb")

    # plot surface temperature distribution per flight
    plot_kmeans_spectra_violin_ts_flight(ds_ea)


def apply_kmeans():
    """
    This function applies K-Means algirithm to the data. It is meant to be
    used also outside of this script. The K-Means is applied on scaled inputs.
    The resulting centroids for each cluster are transformed back again to the
    emissivity space from the scaled space.
    """

    ds_ea = get_data()

    # apply k-means on data with given number of clusters
    k = 4
    km = run_kmeans(n_clusters=k)
    km_y = km.fit_predict(ds_ea.e_scaled.values)
    km_y += 1
    km_dist = km.transform(ds_ea.e_scaled.values)

    ds_ea["km_label"] = ("time", km_y)
    ds_ea.coords["n_cluster"] = np.arange(1, k + 1)
    ds_ea["km_center_scaled"] = (
        ("channel", "n_cluster"),
        km.cluster_centers_.T,
    )
    ds_ea["km_center"] = (
        ("channel", "n_cluster"),
        scale(
            ds_ea.e.values,
            x_scaled=ds_ea["km_center_scaled"].T.values,
            inverse=True,
        ).T,
    )
    ds_ea["km_dist"] = (("time", "n_cluster"), km_dist)

    # relabel clusters
    ds_ea["km_label"] = relabel(ds_ea["km_label"])
    ds_ea.coords["n_cluster"] = [
        RELABEL_DCT[i] for i in ds_ea.n_cluster.values
    ]
    ds_ea = ds_ea.sel(n_cluster=np.sort(ds_ea.n_cluster))

    return ds_ea


def occurrence_statistics(ds_ea):
    """
    Calculates the frequency of occurrence of each class depending on the
    research flight.
    """

    number_per_flight = ds_ea.km_label.groupby(ds_ea.flight_id).count()
    number_per_cluster = ds_ea.km_label.groupby(ds_ea.km_label).count()

    for i in ds_ea.n_cluster.values:
        count = (
            ds_ea.km_label.where(ds_ea.km_label == i)
            .groupby(ds_ea.flight_id)
            .count()
        )
        relative_per_flight = np.round(100 * count / number_per_flight)

        relative_per_class = np.round(
            100 * count / number_per_cluster.sel(km_label=i)
        )

        print(
            f"cluster {i}: {count.flight_id.values} - {count.values} - "
            f"{relative_per_flight.values} - "
            f"{relative_per_class.values}"
        )


def fill_time_gap(ds, dt_max=4):
    """
    Fills measurement gaps with nan's. Otherwise, rectangular stripes may occur
    when plotting radar reflectivity

    Parameters
    ----------
    ds: any xarray dataset with a time dimension
    dt_max: highest tolerated time gap

    Returns
    -------
    ds: same dataset with nans inserted between time gaps
    """

    # creaty copy of the time index
    time_orig = ds.time.values.copy()

    # calculate time stemp in seconds
    dt = (time_orig[1:] - time_orig[:-1]) / np.timedelta64(1, "ns") * 1e-9

    ix = np.argwhere(dt > dt_max).flatten()

    # create times for additional time steps 1 s after (before) start (end) of
    # measurement gap
    time_insert_lower = time_orig[ix] + np.timedelta64(1, "s")
    time_insert_upper = time_orig[ix + 1] - np.timedelta64(1, "s")
    time_insert = np.insert(
        time_insert_upper, np.arange(len(ix)), time_insert_lower
    )

    # insert the additional time steps to original time array and reindex the
    # dataset
    time_reindex = np.insert(time_orig, np.repeat(ix, 2) + 1, time_insert)
    ds = ds.reindex({"time": time_reindex})

    return ds


def on_map():
    """
    View clusters on a map
    """

    fig, axes = plt.subplots(
        2,
        3,
        subplot_kw=dict(projection=ccrs.NorthPolarStereo()),
        sharex=True,
        sharey=True,
    )

    (ax1, ax2, ax3, ax4, ax5, ax6) = axes.flatten()

    ax6.remove()

    for ax in fig.axes:
        ax.set_extent([8.5, 10.5, 80.75, 80.9])

    kwds = dict(vmin=0.5, vmax=1, cmap="magma", transform=ccrs.PlateCarree())

    ax1.set_title("Cluster")
    ax1.scatter(
        ds_ea.lon.sel(channel=9),
        ds_ea.lat.sel(channel=9),
        c=[kmeans_colors[x] for x in ds_ea.km_label.values],
        transform=ccrs.PlateCarree(),
    )

    ax2.set_title("89 GHz")
    ax2.scatter(
        ds_ea.lon.sel(channel=9),
        ds_ea.lat.sel(channel=9),
        c=ds_ea.e.sel(channel=1),
        **kwds,
    )

    ax3.set_title("183 GHz")
    ax3.scatter(
        ds_ea.lon.sel(channel=9),
        ds_ea.lat.sel(channel=9),
        c=ds_ea.e.sel(channel=7),
        **kwds,
    )

    ax4.set_title("243 GHz")
    ax4.scatter(
        ds_ea.lon.sel(channel=9),
        ds_ea.lat.sel(channel=9),
        c=ds_ea.e.sel(channel=8),
        **kwds,
    )

    ax5.set_title("340 GHz")
    ax5.scatter(
        ds_ea.lon.sel(channel=9),
        ds_ea.lat.sel(channel=9),
        c=ds_ea.e.sel(channel=9),
        **kwds,
    )


def plot_scores(n, s, ds_ea):
    """
    This functions plots all three scores distortion, calinski-habarasz, and
    silhouette score.

    Provide all inputs in a list with the order:
    1) distortion
    2) calinski-habarasz
    3) slihouette

    Parameters
    ----------
    n: list of all number of clusters that was tested
    s: list of the scores corresponding to the clusters
    """

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 3.2), gridspec_kw=dict(width_ratios=[1.4, 1])
    )

    twin1 = ax1.twinx()
    twin2 = ax1.twinx()

    twin1.spines["right"].set_visible(True)
    twin2.spines["right"].set_visible(True)

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.2))

    labels = ["Distortion", "Calinski-Habarasz index", "Silhouette score"]
    colors = cmc.bamako(np.linspace(0, 0.75, len(labels)))
    axis = [ax1, twin1, twin2]
    linestyles = ["-", "--", "-."]
    lines = []
    for i in range(len(n)):
        line = axis[i].plot(
            n[i],
            s[i],
            label=labels[i],
            color=colors[i],
            marker=".",
            linestyle=linestyles[i],
        )
        lines.extend(line)

    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Distortion")
    twin1.set_ylabel("Calinski-Habarasz index")
    twin2.set_ylabel("Silhouette score")

    ax1.set_xlim(0.5, 10.5)
    ax1.set_xticks(np.arange(1, 11))

    for i, ax in enumerate(axis):
        ax.yaxis.label.set_color(lines[i].get_color())
        ax.tick_params(axis="y", colors=lines[i].get_color())

    ax1.axvline(x=4, color="gray", linestyle=":")

    ax1.legend(handles=lines, frameon=False, fontsize=8)

    # plot compressed version of all points to the right using pca analysis
    # mark centroids and cluster points as done for four classes
    cmap = mcolors.ListedColormap(kmeans_colors.values())

    # scale features
    x_2d, km_centers_2d = compute_pca(
        ds_ea.e_scaled.values, ds_ea.km_center_scaled.values.T
    )

    # plot voronoi polygons
    ax2 = plot_voronoi(km_centers_2d, ax2)

    ax2.scatter(
        x_2d[:, 0],
        x_2d[:, 1],
        marker=".",
        s=20,
        lw=0,
        alpha=0.5,
        cmap=cmap,
        c=ds_ea.km_label,
    )
    for i, c in enumerate(km_centers_2d):
        ax2.scatter(
            c[0], c[1], marker="$%d$" % (i + 1), alpha=1, s=50, edgecolor="k"
        )

    ax2.set_yticks([])
    ax2.set_xticks([])

    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_aspect("equal")

    ax2.set_xlim(-3.5, 4.5)
    ax2.set_ylim(-4, 4)

    # annotate letters
    for i, ax in enumerate([ax1, ax2]):
        txt = f"({string.ascii_lowercase[i]})"
        ax.annotate(
            txt, xy=(0, 1), xycoords="axes fraction", ha="center", va="bottom"
        )

    write_figure(fig, "kmeans_scores.png")


def plot_flights_2d(ds_ea):
    """
    Visualize the research flights in 2D space
    """

    fig, ax = plt.subplots(1, 1)

    # scale features
    x_2d, km_centers_2d = compute_pca(
        ds_ea.e_scaled.values, ds_ea.km_center_scaled.values.T
    )
    ax.scatter(
        x_2d[:, 0],
        x_2d[:, 1],
        marker=".",
        s=100,
        lw=0,
        cmap="jet",
        c=ds_ea.i_flight_id,
    )
    for i, c in enumerate(km_centers_2d):
        ax.scatter(
            c[0], c[1], marker="$%d$" % (i + 1), alpha=1, s=50, edgecolor="k"
        )

    ax.set_yticks([])
    ax.set_xticks([])

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xcenter = (xlim[0] + xlim[1]) / 2
    ycenter = (ylim[0] + ylim[1]) / 2
    w, h = [xlim[1] - xlim[0], ylim[1] - ylim[0]]
    if w > h:
        ax.set_ylim(ycenter - w / 2, ycenter + w / 2)
    else:
        ax.set_xlim(xcenter - h / 2, xcenter + h / 2)


def get_data():
    """
    Get emissivity data for K-Means classification
    """

    surf_refl = "L"
    channels = [1, 7, 8, 9]

    # read emissivity data
    flight_ids = [
        "ACLOUD_P5_RF23",
        "ACLOUD_P5_RF25",
        "AFLUX_P5_RF08",
        "AFLUX_P5_RF14",
        "AFLUX_P5_RF15",
    ]
    ds_ea, dct_fl = airborne_emissivity_merged(
        flight_ids=flight_ids,
        filter_kwargs=dict(dtb_filter=True, drop_times=True),
    )

    # reduce dataset to specific channels and surface reflection
    ds_ea = ds_ea.sel(surf_refl=surf_refl, channel=channels)

    # match MiRAC-A and MiRAC-P spatially and temporally
    ds_ea = match_miracs(ds_ea)

    # drop times when emissivity is nan in any channel
    ds_ea = ds_ea.sel(time=~ds_ea.e.isnull().any("channel"))

    # scale emissivity
    ds_ea["e_scaled"] = (ds_ea.e.dims, scale(ds_ea.e.values))

    return ds_ea


def closest_to_centroids(ds_ea, v="km_dist"):
    """
    Find observations closest to the cluster centroids

    Parameters
    ----------
    ds_ea: emissivity dataset with kmeans variables
    v: distance measure variable name (e.g. km_dist or db_dist)

    Returns
    -------
    idx: sorted integer index along time axis for every cluster
    """

    idx = ds_ea[v].argsort(axis=0)

    return idx


def view_camera_nearest(ds_ea, fig=None, gs=None, orientation="horizontal"):
    """
    View camera images made at times close to the centroid

    Parameters
    ----------
    ds_ea: emissivity data with clusters
    orientation: vertical or horizontal. Vertical will have the clusters in
    columns and horizontal in rows.
    fig: figure on which to plot the axis
    gs: if provided, the subplots will be created on an existing figure grid.
      I.e. this parameter should contain the gridspec container, in which the
      requires amount of row/col subplots will be created. The figure won't
      be saved in this case.
    """

    idx = closest_to_centroids(ds_ea)

    # number of pictures to show per class
    n_pictures = 8

    if orientation == "vertical":
        n_row = n_pictures
        n_col = len(ds_ea.n_cluster)

        # plot height
        height = 7

    else:
        n_row = len(ds_ea.n_cluster)
        n_col = n_pictures

        # plot height
        height = 4

    dt_max = 0

    height_ratio = X_DIM * n_col / (Y_DIM * n_row)
    width = height_ratio * height

    if gs is None:
        fig, axes = plt.subplots(
            n_row,
            n_col,
            figsize=(width, height),
            gridspec_kw=dict(wspace=0, hspace=0),
        )

    else:
        gs_grid = gs.subgridspec(n_row, n_col, hspace=0, wspace=0)
        axes = []
        for row in range(n_row):
            axes_row = []
            for col in range(n_col):
                ax = fig.add_subplot(
                    gs_grid[row, col],
                    frame_on=False,
                    aspect="equal",
                    xmargin=0,
                    ymargin=0,
                )
                axes_row.append(ax)
            axes.append(axes_row)
        axes = np.array(axes)

    for ax in axes.flat:
        ax.axis("off")
        ax.set_aspect("equal")

    for i_cluster, cluster in enumerate(ds_ea.n_cluster.values):
        if orientation == "vertical":
            i, j = [0, i_cluster]

            # annotation text
            xy = (0.5, 1)
            ha = "center"
            va = "bottom"

        else:
            i, j = [i_cluster, 0]

            # annotation text
            xy = (-0.1, 0.5)
            ha = "right"
            va = "center"

        axes[i, j].annotate(
            f"Cluster {cluster}",
            xy=xy,
            ha=ha,
            va=va,
            xycoords="axes fraction",
            color="k",
        )

        # iterate over the closest samples to centroid until the right number
        # of images is found
        i_axis = 0
        i_closest = 0
        while i_axis < n_pictures:
            # get time and flight id
            ix_time = idx.isel(n_cluster=i_cluster, time=i_closest).item()
            time = ds_ea.time.isel(time=ix_time).values
            flight_id = ds_ea.flight_id.sel(time=time).item()

            if i_closest < 3:
                print(f"The {i_closest} closest time: {time}")

            # read the nearest image
            img, dt, img_time = read_fish_eye(
                time, flight_id, prefix="no_annot_flexible_fish_eye"
            )

            if np.abs(dt) <= dt_max:
                # print information on the image that will be plotted
                print(
                    f"Cluster {cluster}: {i_closest+1} closest image at {time}"
                )

                if orientation == "vertical":
                    axes[i_axis, i_cluster].imshow(img)

                else:
                    axes[i_cluster, i_axis].imshow(img)

                i_axis += 1

            i_closest += 1

    # add label for orange circle being 100 m diameter
    axes[0, 0].annotate(
        "100 m",
        xy=(0.5, 0.7),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        color="coral",
    )

    if gs is None:
        write_figure(fig, "kmeans_images_nearest.png", dpi=600)
        plt.close()


def view_camera_random(ds_ea):
    """
    View random camera images for every cluster
    """

    n_row = len(ds_ea.n_cluster)
    n_col = 6
    dt_max = 0

    height_ratio = X_DIM * n_col / (Y_DIM * n_row)
    height = 4
    width = height_ratio * height

    fig, axes = plt.subplots(n_row, n_col, figsize=(width, height))

    for ax in fig.axes:
        ax.axis("off")
        ax.set_aspect("equal")

    plt.subplots_adjust(wspace=0, hspace=0)

    for i_cluster, cluster in enumerate(ds_ea.n_cluster.values):
        # get times of measurements that belong to cluster
        cluster_times = ds_ea.time.sel(time=ds_ea.km_label == cluster).values
        np.random.shuffle(cluster_times)

        # plot images close to observation time
        i_axis = 0
        i_time = 0
        while i_axis < n_col:
            # get random time and flight id within cluster
            time = cluster_times[i_time]
            flight_id = ds_ea.flight_id.sel(time=time).item()

            # read image
            img, dt, img_time = read_fish_eye(
                time, flight_id, prefix="no_annot_flexible_fish_eye"
            )

            if np.abs(dt) <= dt_max:
                # set image scale depending on flight altitude
                img_width = 250 / (
                    ds_ea.ac_alt.sel(time=i_closest).item() * 1e-3
                )
                x0, x1 = [
                    int(round(X_DIM / 2 - img_width / 2, 0)),
                    int(round(X_DIM / 2 + img_width / 2, 0)),
                ]
                x0, y1 = [
                    int(round(Y_DIM / 2 - img_width / 2, 0)),
                    int(round(Y_DIM / 2 + img_width / 2, 0)),
                ]

                # reduce image dimension
                img = img[x0:x1, y0:y1, :]

                axes[i_cluster, i_axis].imshow(img)
                i_axis += 1

            i_time += 1

    write_figure(fig, "kmeans_images_random.png")

    plt.close()


def match_miracs(ds_ea):
    """
    Match MiRAC-A and MiRAC-P spatially and temporally with the Matching class.
    Note that times are now corresponding to measurement time of MiRAC-P and
    not of MiRAC-A anymore.

    This is done for all surface-sensitive variables. Times, where both
    instruments to not match are dropped.
    """

    # do the matching for all surface- and channel-sensitive variables
    v_lst = []
    for v in list(ds_ea):
        if "channel" in ds_ea[v].coords and "time" in ds_ea[v].coords:
            v_lst.append(v)

    match = Matching.from_dataset(ds_ea, from_channel=1, to_channel=7)
    match.match(threshold=200, temporally=True)

    # select 89 GHz when it intersected with MiRAC-P at current time
    da_ch1 = ds_ea[v_lst].sel(channel=1, time=match.ds_fpr.from_time)

    # set 89 GHz time to the MiRAC-P time
    da_ch1["time"] = ds_ea.time

    # drop times when footprint was too far away
    da_ch1 = da_ch1.sel(time=match.ds_fpr.from_valid.values)
    ds_ea = ds_ea.sel(time=match.ds_fpr.from_valid.values)

    # overwrite emissivity to have now matching footprints at the same time
    ds_ea[v_lst].loc[{"channel": 1}] = da_ch1

    return ds_ea


def plot_quantiles(ds_ea, target_var):
    """
    Prints emissivity quantiles

    Parameters
    ----------
    ds_ea: emissivity dataset
    target_var: target variable (e.g. emissivity or brightness temperature)
    """

    v = [target_var, "km_label"]
    kwds = dict(group="km_label", bins=[-1] + list(np.unique(ds_ea.km_label)))
    q25 = ds_ea[v].groupby_bins(**kwds).quantile(q=0.25)[target_var]
    q75 = ds_ea[v].groupby_bins(**kwds).quantile(q=0.75)[target_var]

    fig, ax = plt.subplots()
    ax.set_title("25 percentile")
    sns.heatmap(q25.round(2).to_pandas(), annot=True, ax=ax, fmt=".2f")

    fig, ax = plt.subplots()
    ax.set_title("75 percentile")
    sns.heatmap(q75.round(2).to_pandas(), annot=True, ax=ax, fmt=".2f")


def run_kmeans(n_clusters, random_state=42):
    """
    Runs K-Means algorithm on sea ice emissivity samples.
    """

    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-6,
        verbose=0,
        random_state=random_state,
        copy_x=True,
        algorithm="lloyd",
    )

    return km


def scale(x_input, x_scaled=None, inverse=False):
    """
    Apply proprocessing standard scaler to data

    Parameters
    ----------
    x: input features of shape (n_samples, n_features
    inverse: scales back the data to the original representation
    x_scaled: if inversed, then provide the data to be inverted here

    Returns
    -------
    x_scaled: scaled input of shape (n_samples, n_features_new)
    """

    scaler = StandardScaler()
    scaler.fit(x_input)

    if not inverse:
        return scaler.transform(x_input)

    else:
        return scaler.inverse_transform(x_scaled)


def compute_pca(x, km_centers):
    """
    Principal component analysis to visualize points along first two EOF.
    Assign a color for each cluster

    Parameters
    ----------
    x: emissivity array (4 channels)
    km_centers: K-Means centroids (for each class)

    Returns
    -------
    x_2d: emissivity array projected along first two EOF
    km_centers: K-Means centroids projected along first two EOF
    """

    pca = PCA(n_components=2)
    pca.fit(x)

    c = pca.components_
    x_2d = x @ c.T
    km_centers_2d = km_centers @ c.T

    return x_2d, km_centers_2d


def elbow_method(x, start=1, stop=10):
    """
    Compute distortions for a range of number of classes.

    Parameters
    ----------
    x: emissivity data of shape (sample, channel)
    start: lowest number of clusters
    stop: highest number of clusters

    Returns
    -------
    distortions: inertia of the clustering
    n_cluster_range: number of clusters
    """

    distortions = np.array([])
    n_cluster_range = np.arange(start, stop + 1)
    for n_clusters in n_cluster_range:
        km = run_kmeans(n_clusters=n_clusters)
        km.fit(x)
        km.labels_ += 1
        distortions = np.append(distortions, km.inertia_)

    return distortions, n_cluster_range


def plot_elbow_method(distortions, n_cluster_range):
    """
    Visualize distortions for a range of number of classes

    Parameters
    ----------
    distortions: inertia of the clustering
    n_cluster_range: number of clusters
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.plot(n_cluster_range, distortions, marker="o")

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Distortion")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    ax.set_xlim(np.min(n_cluster_range) - 0.5, np.max(n_cluster_range) + 0.5)
    ax.set_ylim(bottom=0)

    write_figure(fig, "elbow_method.png")
    plt.close()


def calculate_ch_index(x, start=2, stop=10):
    """
    Calculates the Calinski-Harabasz index for the K-Means clustering with
    varying number of clusters.

    The index is higher when clusters are dense and well separated, which
    relates to a standard concept of a cluster.

    Parameters
    ----------
    x: emissivity data of shape (sample, channel)
    start: lowest number of clusters to test
    stop: highest number of clusters to test

    Returns
    -------
    ch_index: Calinski-Harabasz index
    n_cluster_range: number of clusters
    """

    ch_index = np.array([])
    n_cluster_range = np.arange(start, stop + 1)
    for n_clusters in n_cluster_range:
        print(n_clusters)
        km = run_kmeans(n_clusters=n_clusters)
        km.fit(x)
        km.labels_ += 1
        index = calinski_harabasz_score(x, km.labels_)
        ch_index = np.append(ch_index, index)

    return ch_index, n_cluster_range


def plot_ch_index(n_cluster_range, ch_index):
    """
    Visualize distortions for a range of number of classes

    Parameters
    ----------
    ch_index: Calinski-Harabasz index
    n_cluster_range: number of clusters
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.plot(n_cluster_range, ch_index, marker="o")

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Calinski-Harabasz index")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    ax.set_xlim(np.min(n_cluster_range) - 0.5, np.max(n_cluster_range) + 0.5)
    ax.set_ylim(bottom=0)

    write_figure(fig, "calinski_harabasz_index.png")
    plt.close()


def silhouette_method(x, start=2, stop=10):
    """
    Silhouette method to find optimal number of clusters

    Parameters
    ----------
    x: emissivity data of shape (sample, channel)
    start: lowest number of clusters
    stop: highest number of clusters

    Returns
    -------
    score: silhouette score for each model
    coefficient_dct: silhouette score of samples in each model and cluster
    n_cluster_range: range of clusters (models) from start to stop
    cluster_labels: predictions for each model
    cluster_centers: cluster centers for each model
    """

    n_cluster_range = np.arange(start, stop + 1)
    score = np.array([])
    coefficient_dct = dict()
    cluster_labels = dict()
    cluster_centers = dict()

    for n_clusters in n_cluster_range:
        coefficient_dct[n_clusters] = dict()

        # initialize clusterer
        km = run_kmeans(n_clusters=n_clusters)
        cluster_labels[n_clusters] = km.fit_predict(x)
        cluster_labels[n_clusters] += 1

        cluster_centers[n_clusters] = km.cluster_centers_

        # silhouette_score
        score = np.append(
            score, silhouette_score(x, cluster_labels[n_clusters])
        )

        # silhouette scores for each sample
        score_sample = silhouette_samples(x, cluster_labels[n_clusters])

        # sort samples
        for i in range(n_clusters):
            score_sample_cluster = score_sample[
                cluster_labels[n_clusters] == i + 1
            ]
            score_sample_cluster = np.sort(score_sample_cluster)

            coefficient_dct[n_clusters][i + 1] = score_sample_cluster

    return (
        score,
        coefficient_dct,
        n_cluster_range,
        cluster_labels,
        cluster_centers,
    )


def plot_silhouette_score(n_cluster_range, score):
    """
    Plot silhouette score to find optimal number of classes.

    Parameters
    ----------
    score: silhouette score for each model
    n_cluster_range: range of clusters (models) from start to stop
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(n_cluster_range, score, color="k", marker=".")

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    write_figure(fig, "silhouette_score.png")
    plt.close()


def plot_silhouettes(
    x, score, coefficient_dct, n_cluster_range, cluster_labels, cluster_centers
):
    """
    Silhouette plot and 2D EOF representation of clusters.

    Parameters
    ----------
    x: emissivity data of shape (sample, channel)
    score: silhouette score for each model
    coefficient_dct: silhouette score of samples in each model and cluster
    n_cluster_range: range of clusters (models) from start to stop
    cluster_labels: predictions for each model
    cluster_centers: cluster centers for each model
    """

    # relabel clusters and centroids for class 4
    cluster_labels[4] = relabel(cluster_labels[4])
    cluster_centers[4] = cluster_centers[4][
        np.array(list(RELABEL_DCT.keys())) - 1, :
    ]

    # update silhouette score class labels
    new_dict = {}
    for old_key, new_key in RELABEL_DCT.items():
        new_dict[new_key] = coefficient_dct[4][old_key]
    coefficient_dct[4] = new_dict

    for j, n_clusters in enumerate(n_cluster_range):
        # plot silhouettes and cluster
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # plot silhouettes
        y_lower = 10
        for i in range(n_clusters):
            c = coefficient_dct[n_clusters][i + 1]

            y_upper = y_lower + len(c)

            if n_clusters == 4:
                color = kmeans_colors[i + 1]

            else:
                color = cm.get_cmap("nipy_spectral")(float(i) / n_clusters)

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                c,
                linewidth=0,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # cluster label
            ax1.annotate(
                str(i + 1),
                xy=(0, y_lower + 0.5 * len(c)),
                ha="left",
                va="center",
            )

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # average silhouette score of all values
        ax1.axvline(x=score[j], color="k", linestyle=":")

        # plot the formed clusters using pca
        x_2d, km_centers_2d = compute_pca(x, cluster_centers[n_clusters])

        # plot the reprojected data in 2d plane
        if n_clusters == 4:
            cmap = mcolors.ListedColormap(kmeans_colors.values())
            kwds = dict(cmap=cmap, c=cluster_labels[n_clusters])

        else:
            colors = cm.get_cmap("nipy_spectral")(
                cluster_labels[n_clusters].astype(float) / n_clusters
            )
            kwds = dict(c=colors)

        ax2.scatter(
            x_2d[:, 0], x_2d[:, 1], marker=".", s=30, lw=0, alpha=0.7, **kwds
        )

        for i, c in enumerate(km_centers_2d):
            ax2.scatter(
                c[0],
                c[1],
                marker="$%d$" % (i + 1),
                alpha=1,
                s=50,
                edgecolor="k",
            )

        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster")

        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")

        ax1.set_yticks([])

        ax1.set_xlim(-0.05, 1)
        ax1.set_ylim([0, len(x) + (n_cluster_range.max() + 1) * 10])

        # annotate letters
        for i, ax in enumerate(fig.axes):
            txt = f"({string.ascii_lowercase[i]})"
            ax.annotate(
                txt,
                xy=(1, 1),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=9,
            )

        write_figure(fig, f"silhouette_analysis_{n_clusters}.png")
        plt.close()


def relabel(km_y):
    """
    Change cluster labels (0 lowest emissivity, 3 highest emissivity)

    Parameters
    ----------
    km_y: K-Means predictions
    """

    n_classes = len(np.unique(km_y))
    km_y += n_classes
    for old_class, new_class in RELABEL_DCT.items():
        km_y[km_y == (old_class + n_classes)] = new_class

    return km_y


def plot_kmeans_spectra(ds_ea):
    """
    Visualizes K-Means spectra

    Parameters
    ----------
    ds_ea: airborne emissivity dataset with 'km_label' variable
    """

    fig, ax = plt.subplots()

    for k in ds_ea.n_cluster.values:
        ax.plot(
            ds_ea.center_freq,
            ds_ea.e.sel(time=ds_ea.km_label == k).T,
            color=kmeans_colors[k],
            linewidth=0.05,
            alpha=0.5,
            zorder=0,
        )

        ax.plot(
            ds_ea.center_freq,
            ds_ea.km_center.sel(n_cluster=k),
            color="white",
            linewidth=3,
            marker=".",
            zorder=1,
        )

        ax.plot(
            ds_ea.center_freq,
            ds_ea.km_center.sel(n_cluster=k),
            color=kmeans_colors[k],
            linewidth=2,
            marker=".",
            label=f"cluster {k}",
            zorder=1,
        )

    ax.set_ylim(0.5, 1)
    ax.set_xlim(79, 350)

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Emissivity")

    ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.025))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    ax.legend(
        loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.15, 0)
    )

    write_figure(fig, "kmeans_spectra.png")
    plt.close()


def rgb2gray(r, g, b, offset=-0.3):
    """
    Convert RGB to grayscale and apply an offset to the grayscale value.
    """

    c = 0.299 * r + 0.587 * g + 0.114 * b

    if offset is not None:
        c = c + offset
        if c > 1 or c < 0:
            c = c - 2 * offset

    c = np.array([c, c, c])

    return c


def plot_kmeans_spectra_dist(ds_ea):
    """
    Visualizes K-Means spectra as violin plot. The surface temperature for
    every cluster is shown on a secondary axis

    Parameters
    ----------
    ds_ea: airborne emissivity dataset with 'km_label' variable
    """

    # restructure data
    df = ds_ea.stack({"tc": ("time", "channel")}).to_dataframe()

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.8])
    gs0 = gs[0].subgridspec(1, 2, width_ratios=[1, 0.2])

    ax_e = fig.add_subplot(gs0[0, 0])
    ax_t = fig.add_subplot(gs0[0, 1])

    # plot random images
    view_camera_nearest(ds_ea, fig=fig, gs=gs[1], orientation="horizontal")

    # emissivity
    # sns.violinplot(
    #    x='channel',
    #    y='e',
    #    hue='km_label',
    #    data=df,
    #    palette=kmeans_colors,
    #    linewidth=0.5,
    #    ax=ax_e)

    sns.boxplot(
        x="channel",
        y="e",
        hue="km_label",
        data=df,
        palette=kmeans_colors,
        ax=ax_e,
        fliersize=0.5,
        linewidth=0.75,
        medianprops={"color": "k", "label": "_e_median_"},
    )

    median_colors = [
        rgb2gray(c[0], c[1], c[2]) for c in list(kmeans_colors.values())
    ]
    median_lines = [
        line for line in ax_e.get_lines() if line.get_label() == "_e_median_"
    ]
    for i, line in enumerate(median_lines):
        line.set_color(median_colors[i % len(median_colors)])

    ax_e.set_ylim(0.5, 1)
    ax_e.set_xticklabels(ds_ea.center_freq.values.astype("int"))

    # add vertical line to separate frequencies
    ax_e.tick_params(axis="x", length=0)
    for x in [0.5, 1.5, 2.5]:
        ax_e.axvline(x, color="k", linewidth=0.75)

    ax_e.set_xlabel("Frequency [GHz]")
    ax_e.set_ylabel("Emissivity")

    ax_e.yaxis.set_minor_locator(mticker.MultipleLocator(0.025))
    ax_e.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    ax_e.legend(
        loc="lower left", bbox_to_anchor=(0.3, 0), ncol=1, frameon=False
    )

    # surface temperature
    # sns.violinplot(
    #    x='km_label',
    #    y='ts',
    #    data=df,
    #    palette=kmeans_colors,
    #    linewidth=0.5,
    #    ax=ax_t,
    #    )

    sns.boxplot(
        x="km_label",
        y="ts",
        data=df,
        palette=kmeans_colors,
        ax=ax_t,
        fliersize=0.5,
        linewidth=0.75,
        medianprops={"color": "k", "label": "_ts_median_"},
    )

    median_colors = [
        rgb2gray(c[0], c[1], c[2]) for c in list(kmeans_colors.values())
    ]
    median_lines = [
        line for line in ax_t.get_lines() if line.get_label() == "_ts_median_"
    ]
    for i, line in enumerate(median_lines):
        line.set_color(median_colors[i])

    ax_t.axhline(y=273.15 - 1.8, color="k", linestyle="--", linewidth=0.75)
    ax_t.annotate(
        "-1.8°C",
        xy=(3, 273.15 - 1.8),
        xycoords="data",
        ha="right",
        va="bottom",
    )

    ax_t.set_xlabel("Cluster")
    ax_t.set_ylabel("Surface temperature [K]")

    ax_t.legend().remove()

    # annotate letters
    for i, ax in enumerate(fig.axes[:3]):
        txt = f"({string.ascii_lowercase[i]})"
        ax.annotate(
            txt, xy=(0, 1), xycoords="axes fraction", ha="center", va="bottom"
        )

    write_figure(fig, "kmeans_spectra_dist.png", dpi=600)
    plt.close()


def plot_kmeans_spectra_violin_tb(ds_ea):
    """
    Visualizes TB for each K-Means class and channel as violin plot.

    Parameters
    ----------
    ds_ea: airborne emissivity dataset with 'km_label' variable
    """

    # restructure data
    df = ds_ea.stack({"tc": ("time", "channel")}).to_dataframe()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    sns.violinplot(
        x="channel",
        y="tb",
        hue="km_label",
        data=df,
        palette=kmeans_colors,
        linewidth=0.5,
        ax=ax,
    )

    ax.set_xticklabels(ds_ea.center_freq.values)

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("TB")

    ax.legend(loc="lower right", ncol=4, frameon=False)

    write_figure(fig, "kmeans_spectra_violin_tb.png")
    plt.close()


def plot_kmeans_spectra_violin_ts_flight(ds_ea):
    """
    Visualizes surface temperature for each K-Means class and flight id
    as violin plot.

    Parameters
    ----------
    ds_ea: airborne emissivity dataset with 'km_label' variable
    """

    # restructure data
    df = ds_ea.stack({"tc": ("time", "channel")}).to_dataframe()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    sns.violinplot(
        x="flight_id",
        y="ts",
        hue="km_label",
        data=df,
        palette=kmeans_colors,
        linewidth=0.5,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Surface temperature [K]")

    ax.legend(loc="upper center", ncol=2, frameon=False)

    write_figure(fig, "kmeans_spectra_violin_ts_flight.png")
    plt.close()


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_voronoi(km_centers_2d, ax):
    """
    Plot Voronoi tesselation of K-Means clusters in 2D.

    Parameters
    ----------
    km_centers_2d: K-Means centroids projected along first two EOF
    """

    # compute Voronoi tesselation
    # add points at edges far away to make polygons extent to axis limits
    points = km_centers_2d
    points = np.vstack(
        (
            points,
            [
                [-1000, -1000],
                [+1000, -1000],
                [-1000, +1000],
                [+1000, +1000],
            ],
        )
    )
    vor = Voronoi(points)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # colorize
    for i, region in enumerate(regions[: len(kmeans_colors)], start=1):
        polygon = vertices[region]

        # polygon = vertices[region]
        # ax.plot(*zip(*polygon), alpha=0.4, color=kmeans_colors[i], label=i)
        ax.fill(*zip(*polygon), alpha=0.4, color=kmeans_colors[i], label=i)

    leg = ax.legend()
    leg.set_title("Cluster")

    return ax


if __name__ == "__main__":
    main()
