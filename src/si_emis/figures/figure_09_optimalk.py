"""
Find optimal number of clusters
"""

from si_emis.figures.figure_05_kmeans import (
    apply_kmeans,
    calculate_ch_index,
    elbow_method,
    get_data,
    plot_ch_index,
    plot_elbow_method,
    plot_scores,
    plot_silhouette_score,
    plot_silhouettes,
    silhouette_method,
)


def find_optimal_k():
    """
    Find optimal k by calling silhouette and elbow methods
    """

    ds_ea = get_data()
    x = ds_ea.e_scaled.values
    start = 1
    stop = 10

    # find optimal number of clusters with elbow method
    distortions, n_clusters_d = elbow_method(x=x, start=start, stop=stop)
    plot_elbow_method(distortions, n_clusters_d)

    # find optimal number of clusters with Calinski-Habarasz method
    ch_index, n_clusters_c = calculate_ch_index(
        x=x, start=start + 1, stop=stop
    )
    plot_ch_index(n_clusters_c, ch_index)

    # find optimal number of clusters with silhouette method
    (
        s_score,
        coefficient_dct,
        n_clusters_s,
        cluster_labels,
        cluster_centers,
    ) = silhouette_method(x=x, start=start + 1, stop=stop)
    plot_silhouette_score(n_clusters_s, s_score)
    plot_silhouettes(
        x,
        s_score,
        coefficient_dct,
        n_clusters_s,
        cluster_labels,
        cluster_centers,
    )

    print(s_score)
    print(s_score.mean())

    # make one plot that shows all of the three scores: distortion, calinski-
    # habarasz, and silhouette score
    ds_ea = apply_kmeans()  # for four classes
    plot_scores(
        n=[n_clusters_d, n_clusters_c, n_clusters_s],
        s=[distortions, ch_index, s_score],
        ds_ea=ds_ea,
    )


if __name__ == "__main__":
    find_optimal_k()
