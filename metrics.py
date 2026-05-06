import argparse
import numpy as np
import open3d as o3d


# -------------------------
# Betöltés + mintavételezés
# -------------------------
def read_point_cloud(path, num_points=5000):
    pcd = o3d.io.read_point_cloud(path)

    if len(pcd.points) > num_points:
        idx = np.random.choice(len(pcd.points), num_points, replace=False)
        pcd = pcd.select_by_index(idx)

    return pcd


# -------------------------
# Normalizálás (skála + középpont)
# -------------------------
def normalize(pc):
    points = np.asarray(pc.points)

    center = points.mean(axis=0)
    points = points - center

    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale

    pc.points = o3d.utility.Vector3dVector(points)
    return pc


# -------------------------
# FPFH feature
# -------------------------
def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100
        )
    )

    return fpfh


# -------------------------
# RANSAC + ICP alignment
# -------------------------
def align(source, target):
    voxel_size = 0.01

    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    distance_threshold = voxel_size * 1.5

    print("Running RANSAC...")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 1000)
    )

    source.transform(result.transformation)

    print("Running ICP...")

    threshold = 0.2
    icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    source.transform(icp.transformation)

    return source


# -------------------------
# Chamfer Distance
# -------------------------
def chamfer_distance(pcA, pcB):
    ptsA = np.asarray(pcA.points)
    ptsB = np.asarray(pcB.points)

    treeA = o3d.geometry.KDTreeFlann(pcA)
    treeB = o3d.geometry.KDTreeFlann(pcB)

    distA = []
    for p in ptsB:
        _, _, d = treeA.search_knn_vector_3d(p, 1)
        distA.append(d[0])

    distB = []
    for p in ptsA:
        _, _, d = treeB.search_knn_vector_3d(p, 1)
        distB.append(d[0])

    return np.mean(distA) + np.mean(distB)

def compute_coverage(pc_ref, pc_pred, threshold=0.02):
    """
    pc_ref: ground truth (pl. teljes hajó)
    pc_pred: predikció (pl. részleges scan)

    threshold: mi számít "közelinek"
    """

    pts_ref = np.asarray(pc_ref.points)
    tree = o3d.geometry.KDTreeFlann(pc_pred)

    covered = 0

    for p in pts_ref:
        _, _, d = tree.search_knn_vector_3d(p, 1)

        if d[0] < threshold**2:  # FIGYELEM: Open3D négyzetes távolságot ad!
            covered += 1

    return covered / len(pts_ref)
# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcA", required=True)
    parser.add_argument("--pcB", required=True)
    parser.add_argument("--num_points", type=int, default=10000)

    args = parser.parse_args()

    print("Loading point clouds...")

    pcA = read_point_cloud(args.pcA, args.num_points)
    pcB = read_point_cloud(args.pcB, args.num_points)

    print("Normalizing...")

    pcA = normalize(pcA)
    pcB = normalize(pcB)

    # -------------------------
    # CENTROID IGAZÍTÁS (KRITIKUS!)
    # -------------------------
    print("Center aligning...")

    pcB.translate(pcA.get_center() - pcB.get_center())

    # -------------------------
    # ALIGNMENT
    # -------------------------
    print("Aligning...")

    pcB = align(pcB, pcA)

    # -------------------------
    # VIZUALIZÁCIÓ
    # -------------------------
    print("Visualizing...")

    pcA.paint_uniform_color([1, 0, 0])  # piros
    pcB.paint_uniform_color([0, 1, 0])  # zöld

    o3d.visualization.draw_geometries([pcA, pcB])

    # -------------------------
    # METRIKA
    # -------------------------
    print("Computing Chamfer Distance...")

    dist = chamfer_distance(pcA, pcB)

    print("\nChamfer Distance:")
    print(dist)

    print("Computing coverage...")

    coverage = compute_coverage(pcA, pcB, threshold=0.02)

    print(f"Coverage: {coverage*100:.2f}%")


if __name__ == "__main__":
    main()