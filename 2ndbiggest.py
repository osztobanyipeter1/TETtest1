import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def get_connected_components(points, radius=0.12, min_cluster_size=80):
    if len(points) == 0:
        return np.array([], dtype=int)

    nbrs = NearestNeighbors(radius=radius, algorithm="kd_tree").fit(points)
    graph = nbrs.radius_neighbors_graph(points, mode="connectivity")
    graph = graph.maximum(graph.T)

    _, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    sizes = np.bincount(labels)
    filtered_labels = np.full_like(labels, -1)
    next_label = 0

    for old_label, size in enumerate(sizes):
        if size >= min_cluster_size:
            filtered_labels[labels == old_label] = next_label
            next_label += 1

    return filtered_labels


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("megkoszosabb_merged_25000.ply")

    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    _, inliers = filtered_pcd.segment_plane(
        distance_threshold=0.03, ransac_n=3, num_iterations=2000
    )
    remaining_pcd = filtered_pcd.select_by_index(inliers, invert=True)

    points = np.asarray(remaining_pcd.points)
    labels = get_connected_components(
        points, radius=0.12, min_cluster_size=80
    )

    unique_labels, counts = np.unique(
        labels[labels != -1], return_counts=True
    )

    colors = np.zeros((len(points), 3))
    colors[:] = [0.0, 0.0, 1.0]

    if len(counts) >= 2:
        sorted_indices = np.argsort(-counts)
        second_largest_label = unique_labels[sorted_indices[1]]

        mask = labels == second_largest_label
        colors[mask] = [1.0, 0.5, 0.0]

    remaining_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(
        "2ndkimenet.ply", remaining_pcd, write_ascii=True
    )