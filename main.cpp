/*
    First small pcl project to avoid obstacles on a plane.


*/
#include <iostream>
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <Eigen/Core>
#include <exception>
typedef pcl::PointXYZ PointType;

using namespace std;

//  Parameters

float voxelLeafSize;
float point1, point2;
Eigen::Vector4f cropBoxMinPoint;
Eigen::Vector4f cropBoxMaxPoint;

//

pcl::visualization::PCLVisualizer::Ptr viewPointCloud(pcl::PointCloud<PointType>::Ptr cloud, string title)
{

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0, 0, 0);
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    //viewer->addCoordinateSystem (1.0, "global");
    //viewer->initCameraParameters ();
    viewer->addPointCloud<PointType>(cloud);
    return viewer;
}

pcl::PointCloud<PointType>::Ptr voxelFilter(pcl::PointCloud<PointType>::Ptr cloud)
{
    cout << "\nCloud size before voxel filter: " << cloud->points.size() << "\n";
    pcl::PointCloud<PointType>::Ptr voxelCloud(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> filter;
    filter.setInputCloud(cloud);
    filter.setLeafSize(voxelLeafSize, voxelLeafSize, voxelLeafSize);
    filter.filter(*voxelCloud);
    cout << "Cloud size after voxel filter: " << voxelCloud->points.size() << "\n\n";
    return voxelCloud;
}

pcl::PointCloud<PointType>::Ptr cropBoxFilter(pcl::PointCloud<PointType>::Ptr cloud)
{
    cout << "\nCloud size before crop box filter: " << cloud->points.size() << "\n";
    pcl::CropBox<PointType> region(true);
    pcl::PointCloud<PointType>::Ptr cropBoxCloud(new pcl::PointCloud<PointType>);
    region.setMin(cropBoxMinPoint);
    region.setMax(cropBoxMaxPoint);
    region.setInputCloud(cloud);
    region.filter(*cropBoxCloud);
    cout << "Cloud size after crop box filter: " << cropBoxCloud->points.size() << "\n\n";
    return cropBoxCloud;
}
int main(int argc, char **argv)
{
    try
    {
        stringstream convert{argv[2]};
        convert >> voxelLeafSize;
        stringstream convert1{argv[3]};
        convert1 >> point1;
        cropBoxMinPoint = Eigen::Vector4f(point1, point1, point1, 1);
        stringstream convert2{argv[4]};
        convert2 >> point2;
        cropBoxMaxPoint = Eigen::Vector4f(point2, point2, point2, 1);
    }
    catch (exception &e)
    {
        cerr << "Wrong CL arguments. Terminating.";
        return -1;
    }

    cout << "Launching program...\n\n";
    //Reading the pcd file
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument(argc, argv, "pcd");
    if (!pcd_filename_indices.empty())
    {
        cout << "Loading file..." << endl;
        string filename = argv[pcd_filename_indices[0]];
        if (pcl::io::loadPCDFile(filename, *cloud) == -1)
        {
            cerr << "Was not able to open file \"" << filename << "\".\n";

            return -1;
        }
    }
    else
    {
        cerr << "Need to provide the pcd directory" << endl;
        return -1;
    }
    cout << "Cloud loaded, size: " << cloud->points.size() << endl;

    //Maybe check if XYZRGB and turn it into XYZ

    //Starting the process

    //Voxel filter
    pcl::PointCloud<PointType>::Ptr voxelCloud = voxelFilter(cloud);
    //Crop box filter
    pcl::PointCloud<PointType>::Ptr cropBoxCloud = cropBoxFilter(voxelCloud);

    //Visualization
    cout << "Visualizing ...\n";
    pcl::visualization::PCLVisualizer::Ptr viewer = viewPointCloud(cloud, "Original Point Cloud");
    pcl::visualization::PCLVisualizer::Ptr viewer2 = viewPointCloud(voxelCloud, "Voxel Filter");
    pcl::visualization::PCLVisualizer::Ptr viewer3 = viewPointCloud(cropBoxCloud, "Crop Box Filter");
    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
    cout << "\nDone visualizing\n\n";

    cout << "Program ended" << endl;
    return 0;
}
