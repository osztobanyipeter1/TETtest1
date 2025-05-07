#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>

int main()
{
    // Betöltjük a pontfelhőt a centered.ply fájlból
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("centered.ply", *cloud) == -1)
    {
        PCL_ERROR("Nem sikerült betölteni a 'centered.ply' fájlt.\n");
        return -1;
    }

    // Normálvektor becslés
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    tree->setInputCloud(cloud);
    normal_est.setInputCloud(cloud);
    normal_est.setSearchMethod(tree);
    normal_est.setKSearch(20);
    normal_est.compute(*normals);

    // XYZ + normál összefűzése
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // Keresési fa a GP3 algoritmushoz
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud(cloud_with_normals);

    // GP3 beállítás
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh mesh;

    gp3.setSearchRadius(0.025);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4);     // 45 fok
    gp3.setMinimumAngle(M_PI / 18);           // 10 fok
    gp3.setMaximumAngle(2 * M_PI / 3);        // 120 fok
    gp3.setNormalConsistency(false);

    // Rekonstrukció
    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree2);
    gp3.reconstruct(mesh);

    // Megjelenítés
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("GP3 Mesh"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPolygonMesh(mesh, "meshed surface");
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
