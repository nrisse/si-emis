"""
Plot camera images on 2D PCA plane of emissivity space. The frame of the
colors is the K-Means class.
"""

import matplotlib.pyplot as plt
import numpy as np
from lizard.writers.figure_to_file import write_figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from si_emis.figures.figure_05_kmeans import (
    apply_kmeans,
    compute_pca,
    plot_voronoi,
)
from si_emis.readers.fish_eye import read_fish_eye


def main():
    """
    Plot images in 2D space.
    """

    # apply k-means clustering
    ds_ea = apply_kmeans()

    plot_images_2d(ds_ea)


def plot_images_2d(ds_ea):
    """
    Plot sea ice images inside the 2D space
    """

    fig, ax = plt.subplots(1, 1)

    x_2d, km_centers_2d = compute_pca(
        ds_ea.e_scaled.values, ds_ea.km_center_scaled.values.T
    )

    # loop over every point, read image at same time and plot it small
    # inside the main plot
    indices = np.arange(0, x_2d.shape[0])
    np.random.shuffle(indices)
    for i, index in enumerate(indices):
        print(f"Progress: {i / len(indices) * 100:.1f}")

        # read camera image
        time = ds_ea.time[index].values
        flight_id = ds_ea.flight_id.isel(time=index).item()

        # read image
        img, dt, img_time = read_fish_eye(
            time, flight_id, prefix="no_annot_flexible_fish_eye"
        )

        if np.abs(dt) == 0:
            # plot image
            imagebox = OffsetImage(img, zoom=0.0075)
            imagebox.image.axes = ax

            ab = AnnotationBbox(
                imagebox,
                x_2d[index, :],
                xycoords="data",
                frameon=False,
                box_alignment=(0.5, 0.5),
                bboxprops=dict(alpha=0),
                zorder=3,
            )
            ax.add_artist(ab)

    # plot voronoi polygons
    ax = plot_voronoi(km_centers_2d, ax)

    # centroids
    for i, c in enumerate(km_centers_2d):
        ax.scatter(
            c[0],
            c[1],
            marker="$%d$" % (i + 1),
            alpha=1,
            s=50,
            edgecolor="k",
            zorder=5,
        )

    ax.set_yticks([])
    ax.set_xticks([])

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")

    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-4, 4)

    write_figure(fig, "kmeans_images_2d.png", dpi=900)


if __name__ == "__main__":
    main()
