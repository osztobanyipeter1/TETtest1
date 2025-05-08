#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

int main()
{
    // Load input file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("centered.ply", *cloud) == -1) {
        PCL_ERROR("Couldn't read file centered.ply \n");
        return -1;
    }

    std::cout << "Loaded " << cloud->size() << " points from centered.ply" << std::endl;

    // Optional: Downsample the point cloud to speed up processing
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(0.02f, 0.02f, 0.02f); // Adjust based on your point cloud density
    voxel_grid.filter(*cloud_downsampled);
    
    std::cout << "Downsampled to " << cloud_downsampled->size() << " points" << std::endl;

    // Estimate normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    
    tree->setInputCloud(cloud_downsampled);
    normal_estimator.setInputCloud(cloud_downsampled);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setKSearch(20); // Adjust based on your point cloud density
    normal_estimator.compute(*normals);

    // Concatenate the XYZ and normal fields
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud_downsampled, *normals, *cloud_with_normals);

    // Create search tree for points with normals
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud(cloud_with_normals);

    // Initialize Greedy Projection Triangulation
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set parameters
    gp3.setSearchRadius(0.01); // Adjust based on your point cloud density (maximum edge length)
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(500);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Perform reconstruction
    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree2);
    gp3.reconstruct(triangles);

    // Save the mesh
    pcl::io::savePLYFile("output_mesh.ply", triangles);

    // Visualize the result
    pcl::visualization::PCLVisualizer viewer("Surface Reconstruction");
    viewer.addPolygonMesh(triangles, "mesh");
    
    // Add original point cloud for comparison
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    viewer.addPointCloud<pcl::PointXYZ>(cloud, single_color, "original cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original cloud");

    std::cout << "Press r to centre and zoom the viewer so that the entire cloud is visible" << std::endl;
    std::cout << "Press q to exit the viewer" << std::endl;

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }

    return 0;
}