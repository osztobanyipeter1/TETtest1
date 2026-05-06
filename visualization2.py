import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def save_cloud(pcd, filename):
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


def colorize_clusters(pcd, labels):
    labels = np.asarray(labels)

    pcd_col = o3d.geometry.PointCloud(pcd)

    if len(labels) == 0 or len(pcd.points) == 0:
        pcd_col.paint_uniform_color([0.5, 0.5, 0.5])
        return pcd_col

    colors = np.zeros((len(labels), 3), dtype=np.float64)
    unique_labels = np.unique(labels)
    valid_labels = [l for l in unique_labels if l != -1]

    if len(valid_labels) == 0:
        colors[:] = [0.5, 0.5, 0.5]
    else:
        cmap = plt.get_cmap("tab20")
        for i, label in enumerate(labels):
            if label == -1:
                colors[i] = [0.2, 0.2, 0.2]
            else:
                colors[i] = cmap(label % 20)[:3]

    pcd_col.colors = o3d.utility.Vector3dVector(colors)
    return pcd_col


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return filtered


def iterative_plane_removal(
    pcd,
    distance_threshold=0.5,
    ransac_n=10,
    num_iterations=20,
    min_plane_points=200,
    max_planes=5
):
    remaining = o3d.geometry.PointCloud(pcd)
    plane_clouds = []

    for _ in range(max_planes):
        if len(remaining.points) < min_plane_points:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        if len(inliers) < min_plane_points:
            break

        plane = remaining.select_by_index(inliers)
        plane_clouds.append((plane_model, plane))
        remaining = remaining.select_by_index(inliers, invert=True)

    return remaining, plane_clouds


def cluster_dbscan(pcd, eps=0.15, min_points=500):
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )
    return labels


def remove_small_clusters(pcd, labels, min_cluster_size=100):
    labels = np.asarray(labels)

    if len(labels) == 0 or len(pcd.points) == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int)

    valid_labels = []
    for label in np.unique(labels):
        if label == -1:
            continue
        size = np.sum(labels == label)
        if size >= min_cluster_size:
            valid_labels.append(label)

    if len(valid_labels) == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int)

    keep_idx = [i for i, l in enumerate(labels) if l in valid_labels]
    filtered = pcd.select_by_index(keep_idx)
    new_labels = labels[keep_idx]

    return filtered, new_labels


def merge_planes(plane_clouds):
    merged = o3d.geometry.PointCloud()
    colors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]

    for i, (_, plane) in enumerate(plane_clouds):
        plane_col = o3d.geometry.PointCloud(plane)
        plane_col.paint_uniform_color(colors[i % len(colors)])
        merged += plane_col

    return merged


def print_cluster_stats(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        print("Nincs címke.")
        return

    unique = np.unique(labels)
    valid = [l for l in unique if l != -1]
    print(f"Klaszterek száma: {len(valid)}")
    print(f"Zajpontok száma: {np.sum(labels == -1)}")

    for l in valid:
        print(f"  - klaszter {l}: {np.sum(labels == l)} pont")


def main():
    input_file = "/home/buvr_tp4/Desktop/ALLPLYS/FPS/koszos_merged_25000.ply"

    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Eredeti pontszám: {len(pcd.points)}")

    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    print(f"Voxel downsample után: {len(pcd.points)}")

    pcd_filtered = remove_outliers(
        pcd,
        nb_neighbors=20,
        std_ratio=2.0
    )
    print(f"Outlier removal után: {len(pcd_filtered.points)}")

    remaining, plane_clouds = iterative_plane_removal(
        pcd_filtered,
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=20,
        min_plane_points=4000,
        max_planes=5
    )

    print(f"Talált síkok száma: {len(plane_clouds)}")
    for i, (model, plane) in enumerate(plane_clouds):
        a, b, c, d = model
        print(f"Sík {i}: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0, pontok: {len(plane.points)}")

    print(f"Maradék pontok száma: {len(remaining.points)}")

    labels = cluster_dbscan(
        remaining,
        eps=0.5,
        min_points=300
    )

    print_cluster_stats(labels)

    clustered_colored = colorize_clusters(remaining, labels)

    large_clusters, large_labels = remove_small_clusters(
        remaining,
        labels,
        min_cluster_size=80
    )
    large_clusters_colored = colorize_clusters(large_clusters, large_labels)

    planes_merged = merge_planes(plane_clouds)

    save_cloud(pcd, "01_downsampled.ply")
    save_cloud(pcd_filtered, "02_filtered.ply")
    save_cloud(planes_merged, "03_detected_planes.ply")
    save_cloud(remaining, "04_remaining_after_plane_removal.ply")
    save_cloud(clustered_colored, "05_dbscan_clusters_colored.ply")

    if len(large_clusters.points) > 0:
        save_cloud(large_clusters_colored, "06_large_clusters_only.ply")

    print("Mentett fájlok:")
    print("- 01_downsampled.ply")
    print("- 02_filtered.ply")
    print("- 03_detected_planes.ply")
    print("- 04_remaining_after_plane_removal.ply")
    print("- 05_dbscan_clusters_colored.ply")
    if len(large_clusters.points) > 0:
        print("- 06_large_clusters_only.ply")

    o3d.visualization.draw_geometries([pcd], window_name="01 Downsampled")
    o3d.visualization.draw_geometries([pcd_filtered], window_name="02 Filtered")
    if len(planes_merged.points) > 0:
        o3d.visualization.draw_geometries([planes_merged], window_name="03 Detected Planes")
    o3d.visualization.draw_geometries([remaining], window_name="04 Remaining")
    o3d.visualization.draw_geometries([clustered_colored], window_name="05 DBSCAN Clusters")
    if len(large_clusters.points) > 0:
        o3d.visualization.draw_geometries([large_clusters_colored], window_name="06 Large Clusters Only")


if __name__ == "__main__":
    main()
