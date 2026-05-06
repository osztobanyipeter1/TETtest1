import open3d as o3d

mesh = o3d.io.read_triangle_mesh("small_ship.STL")
mesh.compute_vertex_normals()

pcd = mesh.sample_points_uniformly(number_of_points=200000)

o3d.io.write_point_cloud("small_ship200000.ply", pcd)