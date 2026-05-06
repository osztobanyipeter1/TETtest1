import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def save_cloud(pcd, filename):
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


def colorize_labels(pcd, labels):
    labels = np.asarray(labels)
    pcd_col = o3d.geometry.PointCloud(pcd)

    if len(labels) == 0 or len(pcd.points) == 0:
        pcd_col.paint_uniform_color([0.5, 0.5, 0.5])
        return pcd_col

    colors = np.zeros((len(labels), 3), dtype=np.float64)
    cmap = plt.get_cmap("tab20")

    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l != -1]

    if len(valid_labels) == 0:
        colors[:] = [0.5, 0.5, 0.5]
    else:
        for i, label in enumerate(labels):
            if label == -1:
                colors[i] = [0.15, 0.15, 0.15]
            else:
                colors[i] = cmap(int(label) % 20)[:3]

    pcd_col.colors = o3d.utility.Vector3dVector(colors)
    return pcd_col


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return filtered


def remove_largest_plane(
    pcd,
    distance_threshold=0.03,
    ransac_n=3,
    num_iterations=2000,
    min_plane_points=1000
):
    if len(pcd.points) < min_plane_points:
        return pcd, o3d.geometry.PointCloud(), None

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    if len(inliers) < min_plane_points:
        return pcd, o3d.geometry.PointCloud(), None

    plane_cloud = pcd.select_by_index(inliers)
    remaining_cloud = pcd.select_by_index(inliers, invert=True)
    return remaining_cloud, plane_cloud, plane_model


def connected_component_clustering(points, radius=0.12, min_cluster_size=50):
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)

    nbrs = NearestNeighbors(radius=radius, algorithm="kd_tree")
    nbrs.fit(points)

    graph = nbrs.radius_neighbors_graph(points, mode="connectivity")
    graph = graph.maximum(graph.T)

    n_components, labels = connected_components(
        csgraph=graph,
        directed=False,
        return_labels=True
    )

    sizes = np.bincount(labels)
    filtered_labels = labels.copy()

    next_label = 0
    remap = {}

    for old_label, size in enumerate(sizes):
        if size < min_cluster_size:
            remap[old_label] = -1
        else:
            remap[old_label] = next_label
            next_label += 1

    for i in range(len(filtered_labels)):
        filtered_labels[i] = remap[filtered_labels[i]]

    return filtered_labels


def keep_only_selected_clusters(pcd, labels, selected_clusters=None):
    labels = np.asarray(labels)

    if len(labels) == 0 or len(pcd.points) == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int)

    if selected_clusters is None:
        valid = [l for l in np.unique(labels) if l != -1]
    else:
        valid = selected_clusters

    keep_idx = [i for i, l in enumerate(labels) if l in valid]

    if len(keep_idx) == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int)

    filtered = pcd.select_by_index(keep_idx)
    new_labels = labels[keep_idx]
    return filtered, new_labels


def print_cluster_stats(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        print("Nincs klasztercímke.")
        return

    unique = np.unique(labels)
    valid = [l for l in unique if l != -1]

    print(f"Klaszterek száma: {len(valid)}")
    print(f"Zajpontok száma: {np.sum(labels == -1)}")

    for l in valid:
        print(f"  - klaszter {l}: {np.sum(labels == l)} pont")


def main():
    #input_file = "/home/buvr_tp4/Desktop/ALLPLYS/koszos_merged.ply"
    input_file = "koszos_merged_500000.ply"

    USE_OUTLIER_REMOVAL = True
    NB_NEIGHBORS = 20
    STD_RATIO = 2.0

    REMOVE_PLANE = True
    PLANE_DISTANCE_THRESHOLD = 0.03
    PLANE_RANSAC_N = 3
    PLANE_NUM_ITER = 2000
    PLANE_MIN_POINTS = 1000

    CONNECT_RADIUS = 0.12
    MIN_CLUSTER_SIZE = 80

    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Eredeti pontszám: {len(pcd.points)}")

    if USE_OUTLIER_REMOVAL:
        pcd = remove_outliers(
            pcd,
            nb_neighbors=NB_NEIGHBORS,
            std_ratio=STD_RATIO
        )
        print(f"Outlier removal után: {len(pcd.points)}")

    plane_cloud = o3d.geometry.PointCloud()
    plane_model = None

    if REMOVE_PLANE:
        pcd, plane_cloud, plane_model = remove_largest_plane(
            pcd,
            distance_threshold=PLANE_DISTANCE_THRESHOLD,
            ransac_n=PLANE_RANSAC_N,
            num_iterations=PLANE_NUM_ITER,
            min_plane_points=PLANE_MIN_POINTS
        )

        if plane_model is not None:
            a, b, c, d = plane_model
            print(f"Eltávolított sík: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            print(f"Sík pontjai: {len(plane_cloud.points)}")
        else:
            print("Nem talált elég nagy síkot.")

    points = np.asarray(pcd.points)
    labels = connected_component_clustering(
        points,
        radius=CONNECT_RADIUS,
        min_cluster_size=MIN_CLUSTER_SIZE
    )

    print_cluster_stats(labels)

    clustered_colored = colorize_labels(pcd, labels)

    # A 2 legnagyobb klaszter kiválasztása helyett csak a második legnagyobb
    unique_labels = np.unique(labels)
    valid_clusters = [l for l in unique_labels if l != -1]

    # Klaszterek méretének meghatározása
    cluster_sizes = [(l, np.sum(labels == l)) for l in valid_clusters]
    # Rendezés méret szerint csökkenő sorrendben
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    # Csak a második legnagyobb klaszter kiválasztása (ha van legalább 2)
    if len(cluster_sizes) >= 2:
        second_largest_cluster = [cluster_sizes[1][0]]
        #second_largest_cluster = [cluster_sizes[1][0],cluster_sizes[2][0]]
        print(f"A 2. legnagyobb klaszter: {cluster_sizes[1][0]} (méret: {cluster_sizes[1][1]} pont)")
        print(f"Az 1. legnagyobb klaszter: {cluster_sizes[0][0]} (méret: {cluster_sizes[0][1]} pont) - NEM jelenik meg")
    else:
        second_largest_cluster = []
        print(f"Nincs második legnagyobb klaszter (összesen {len(cluster_sizes)} klaszter van)")

    kept_cloud, kept_labels = keep_only_selected_clusters(
        pcd,
        labels,
        selected_clusters=second_largest_cluster if second_largest_cluster else None
    )

    kept_colored = colorize_labels(kept_cloud, kept_labels)

    save_cloud(pcd, "01_remaining_input.ply")
    if len(plane_cloud.points) > 0:
        plane_cloud.paint_uniform_color([1.0, 0.0, 0.0])
        save_cloud(plane_cloud, "02_removed_plane.ply")
    save_cloud(clustered_colored, "03_connected_components_colored.ply")
    if len(kept_cloud.points) > 0:
        save_cloud(kept_colored, "04_large_connected_components_only.ply")

    print("Mentett fájlok:")
    print("- 01_remaining_input.ply")
    if len(plane_cloud.points) > 0:
        print("- 02_removed_plane.ply")
    print("- 03_connected_components_colored.ply")
    if len(kept_cloud.points) > 0:
        print("- 04_large_connected_components_only.ply")

    o3d.visualization.draw_geometries([pcd], window_name="Remaining input")
    if len(plane_cloud.points) > 0:
        o3d.visualization.draw_geometries([plane_cloud], window_name="Removed plane")
    o3d.visualization.draw_geometries([clustered_colored], window_name="Connected components")
    if len(kept_cloud.points) > 0:
        o3d.visualization.draw_geometries([kept_colored], window_name="Large connected components only")


if __name__ == "__main__":
    main()
