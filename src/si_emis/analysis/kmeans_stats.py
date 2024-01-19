"""
Prints Median values of K-Means clusters
"""

from si_emis.figures.figure_05_kmeans import apply_kmeans


def emissivity_statistics():
    """
    Prints quantiles
    """

    # apply kmeans
    ds_ea = apply_kmeans()

    da_q25 = ds_ea.e.groupby(ds_ea.km_label).quantile(q=0.25)
    da_q50 = ds_ea.e.groupby(ds_ea.km_label).quantile(q=0.50)
    da_q75 = ds_ea.e.groupby(ds_ea.km_label).quantile(q=0.75)

    for cluster in da_q25.km_label.values:
        for channel in da_q25.channel.values:
            q25 = round(
                da_q25.sel(km_label=cluster, channel=channel).item(), 2
            )
            q50 = round(
                da_q50.sel(km_label=cluster, channel=channel).item(), 2
            )
            q75 = round(
                da_q75.sel(km_label=cluster, channel=channel).item(), 2
            )
            print(
                f"Cluster {cluster}, channel {channel}: {q50} / {q25} - {q75}"
            )

    # amount of emissivities above 1
    ds_ea_h1 = ds_ea.sel(
        time=ds_ea.e.sel(channel=1).reset_coords(drop=True) > 1
    )
    ds_ea_h1.km_label.groupby(ds_ea_h1.km_label).count()


if __name__ == "__main__":
    emissivity_statistics()
