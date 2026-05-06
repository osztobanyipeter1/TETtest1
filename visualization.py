import open3d as o3d

# PLY fájl beolvasása
pcd = o3d.io.read_point_cloud("centered.ply")

# Információ kiírása
print(pcd)
print("Pontok száma:", len(pcd.points))

# Megjelenítés
o3d.visualization.draw_geometries([pcd])