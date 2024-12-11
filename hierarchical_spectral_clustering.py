import numpy as np
import networkx as nx
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans


def nested_spectral_clustering(network, 
                               mean_affinity_target=0.55,
                               max_component=20,
                               select_biggest_cc=False,
                               select_cc_node=None):
    """
    Perform a nested spectral clustering algorithm on a network.

    Parameters:
    -----------
    network : networkx.Graph
        The input network graph.

    mean_affinity_target : float, optional, default=56
        The target mean affinity for stopping cluster subdivision.

    max_component : int, optional, default=20
        The maximum number of components to use in spectral embedding.

    select_biggest_cc : bool, optional, default=False
        If True and the graph has multiple connected components (CCs),
        selects the largest connected component for clustering.

    select_cc_node : any, optional, default=None
        If specified and the graph has multiple CCs, selects the CC
        containing the specified node for clustering.

    Returns:
    --------
    final_clusters : list of np.ndarray
        A list of arrays, where each array contains the node indices
        of a final cluster.

    outliers : list of np.ndarray
        A list of arrays, where each array contains the node indices
        of clusters classified as outliers.

    Raises:
    -------
    ValueError
        If the graph has multiple CCs but no strategy to select one is
        provided (via select_biggest_cc or select_cc_node).
    """
    # Handle graphs with multiple connected components (CCs)
    if nx.number_connected_components(network) > 1:
        if select_cc_node:
            # Select the CC containing the specified node
            new_net = None
            for c in nx.connected_components(network):
                if select_cc_node in c:
                    new_net = network.subgraph(c)
                    print(f"Working only on the connected component containing node {select_cc_node}")
                    break
            if new_net is None:
                raise ValueError(f"Node {select_cc_node} not found in graph nodes. Example node labels: {list(network.nodes())}")
        elif select_biggest_cc:
            # Select the largest CC
            new_net = network.subgraph(sorted(nx.connected_components(network), key=len, reverse=True)[0])
        else:
            raise ValueError("Found multiple connected components but no option to select a given connected component (select_biggest_cc or select_cc_node)")
        network = new_net

    # Convert the network to a weighted adjacency matrix (affinity matrix)
    affinity = nx.to_numpy_array(network, weight="score")

    # Set self-affinity (diagonal values) to a high value to avoid splitting single nodes
    np.fill_diagonal(affinity, 1)

    # Initialize clusters and outliers
    final_clusters = []
    current_clusters = [np.arange(len(network.nodes()))]
    outliers = []

    # Iteratively refine clusters
    while len(current_clusters):
        previous_clusters = current_clusters
        current_clusters = []
        for cluster in previous_clusters:
            cluster_affinity = affinity[np.ix_(cluster, cluster)]

            # Handle edge cases for small or well-clustered groups
            if len(cluster) == 1:
                outliers.append(cluster)
                continue
            elif cluster_affinity.mean() >= mean_affinity_target:
                final_clusters.append(cluster)
                continue
            elif len(cluster) <= 2:
                outliers.append(cluster)
                continue

            # Perform spectral embedding for further clustering
            se = SpectralEmbedding(n_components=min(max_component, len(cluster)), affinity='precomputed')
            X_transformed = se.fit_transform(cluster_affinity)

            # Compute eigenvalues and find the optimal number of clusters
            eigenvalues = reconstruct_eigenvalues(cluster_affinity, se.embedding_)
            relative_eigengaps = (eigenvalues[2:] - eigenvalues[1:-1]) / eigenvalues[1:-1]
            if len(relative_eigengaps):
                optimal_n_clusters = max(2, np.argmax(relative_eigengaps) + 1)
            else:
                optimal_n_clusters = 2

            # Perform clustering using k-means
            se = SpectralEmbedding(n_components=optimal_n_clusters, affinity='precomputed')
            X_transformed = se.fit_transform(cluster_affinity)
            kmeans = KMeans(n_clusters=optimal_n_clusters)
            labels = kmeans.fit_predict(X_transformed)

            # Assign nodes to new clusters
            for lab in np.unique(labels):
                current_clusters.append(cluster[labels == lab])

    return final_clusters, outliers
