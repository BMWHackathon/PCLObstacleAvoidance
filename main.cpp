/*
    First small pcl project to avoid obstacles on a plane.


*/
#include <boost/filesystem.hpp>

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

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <chrono>

typedef pcl::PointXYZ PointType;

using namespace boost::filesystem;
using namespace std;

//  Parameters

float voxelLeafSize = 0.25;                                                                 //0.2
float point1x = -11, point1y = -6, point1z = -11, point2x = 19, point2y = 7, point2z = 19; //-11 -6 -11    19 7 19
Eigen::Vector4f cropBoxMinPoint = Eigen::Vector4f(point1x, point1y, point1z, 1);
Eigen::Vector4f cropBoxMaxPoint = Eigen::Vector4f(point2x, point2y, point2z, 1);
int frame = 1;
bool dev = false;
//

pcl::visualization::PCLVisualizer::Ptr viewPointCloud(pcl::PointCloud<PointType>::Ptr cloud, string title, Eigen::Vector3f color)
{

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0, 0, 0);
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    //viewer->addCoordinateSystem (1.0, "global");
    //viewer->initCameraParameters();
    viewer->addPointCloud<PointType>(cloud, title);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], title);
    return viewer;
}

void addPointCloudToViewer(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<PointType>::Ptr cloud, string id, Eigen::Vector3f color)
{
    viewer->addPointCloud<PointType>(cloud, id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], id);
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

pair<typename pcl::PointCloud<PointType>::Ptr, typename pcl::PointCloud<PointType>::Ptr> RANSACsegmentation(pcl::PointCloud<PointType>::Ptr cloud)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    cout << "Segmenting..." << endl;
    pcl::SACSegmentation<PointType> seg;
    seg.setOptimizeCoefficients(true);     // Enable model coefficient refinement (optional).
    seg.setInputCloud(cloud);              //input
    seg.setModelType(pcl::SACMODEL_PLANE); // Configure the object to look for a plane.
    seg.setMethodType(pcl::SAC_RANSAC);    // Use RANSAC method.
    seg.setMaxIterations(10000);           //Maximum number of executions
    seg.setDistanceThreshold(0.1);         // Distance information to be processed as inlier // Set the maximum allowed distance to the model.
    //seg.setRadiusLimits(0, 0.1);     // cylinder, Set minimum and maximum radii of the cylinder.
    seg.segment(*inliers, *coefficients);

    pcl::PointCloud<PointType>::Ptr plane(new pcl::PointCloud<PointType>),
        objects(new pcl::PointCloud<PointType>);
    pcl::copyPointCloud<PointType>(*cloud, *inliers, *plane);

    pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true); //false
    extract.filter(*objects);
    cout << "Objects cloud size: " << objects->points.size() << endl;
    cout << "Plane cloud size: " << plane->points.size() << endl;
    pair<typename pcl::PointCloud<PointType>::Ptr, typename pcl::PointCloud<PointType>::Ptr> segResult(plane, objects);
    return segResult;
}

pcl::PointCloud<PointType>::Ptr noiseRemoval(pcl::PointCloud<PointType>::Ptr cloud)
{
    cout << "Objects size before noise removal: " << cloud->points.size() << endl;
    pcl::StatisticalOutlierRemoval<PointType> sor;
    pcl::PointCloud<PointType>::Ptr noiseFreeObjects(new pcl::PointCloud<PointType>);
    sor.setInputCloud(cloud);      //input
    sor.setMeanK(50);              //Neighboring points considered during analysis (50)
    sor.setStddevMulThresh(1.0);   // Distance information to be processed by outlier
    sor.filter(*noiseFreeObjects); // Apply filter
    cout << "Objects size after noise removal: " << noiseFreeObjects->points.size() << endl;
    return noiseFreeObjects;
}

vector<pcl::PointCloud<PointType>::Ptr> extractClusters(pcl::PointCloud<PointType>::Ptr cloud)
{

    cout << "\nExtracting clusters...\n";
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    tree->setInputCloud(cloud);

    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance(0.8);
    ec.setMinClusterSize(30);
    ec.setMaxClusterSize(125000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    vector<pcl::PointCloud<PointType>::Ptr> clusters;
    int j = 0;
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);
        for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud->points[*pit]); //*
        cout << "Cluster " << (j + 1) << " size: " << cloud_cluster->points.size() << endl;
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        clusters.push_back(cloud_cluster);
        //View clusters:
        // pcl::visualization::PCLVisualizer viewer("PCL Viewer");
        // viewer.addPointCloud<pcl::PointXYZ>(cloud_cluster);
        // while (!viewer.wasStopped())
        // {
        //     viewer.spinOnce();
        // }
        j++;
    }
    cout << "\n"
         << j << " clusters extracted" << endl;

    return clusters;
}

void createBoundingBoxes(vector<pcl::PointCloud<PointType>::Ptr> clusters, pcl::visualization::PCLVisualizer::Ptr viewer)
{
    cout << "\nCreating bounding rectangles for clusters...\n";
    for (size_t i = 0; i < clusters.size(); i++)
    {
        PointType minPoint, maxPoint;
        pcl::getMinMax3D(*(clusters[i]), minPoint, maxPoint);

        string id = "Cluster " + (i + 1);

        viewer->addCube(minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z, 1.0, 0, 0, id);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
        // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0, id);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1.0, id);
        string idFill = "Cluster Fill " + (i + 1);
        viewer->addCube(minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z, 1.0, 0, 0, idFill);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, idFill);
        //viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0, idFill);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, idFill);
    }

    //return viewer;
}

int main(int argc, char **argv)
{
    cout << "Select the mode: (1) Dev (Work with one frame only), (2) Entire sequence: ";
    int mode;
    cin >> mode;
    if (mode == 2)
    {
        dev = false;
    }

    cout << "\n\nLaunching program...\n\n";
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
    if (dev)
    {
        try
        {
            stringstream convert{argv[2]};
            convert >> voxelLeafSize;

            stringstream convert1x{argv[3]};
            convert1x >> point1x;
            stringstream convert1y{argv[4]};
            convert1y >> point1y;
            stringstream convert1z{argv[5]};
            convert1z >> point1z;
            cropBoxMinPoint = Eigen::Vector4f(point1x, point1y, point1z, 1);

            stringstream convert2x{argv[6]};
            convert2x >> point2x;
            stringstream convert2y{argv[7]};
            convert2y >> point2y;
            stringstream convert2z{argv[8]};
            convert2z >> point2z;
            cropBoxMaxPoint = Eigen::Vector4f(point2x, point2y, point2z, 1);
        }
        catch (exception &e)
        {
            cerr << "Wrong CL arguments. Terminating.";
            return -1;
        }

        //Reading the pcd file

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

    } //End if dev

    path p("../../resources/data_1");
    cout << "Getting files: " << endl;
    vector<string> files;
    for (auto i = directory_iterator(p); i != directory_iterator(); i++)
    {
        if (!is_directory(i->path())) //we eliminate directories
        {
            files.push_back(i->path().filename().string());
        }
        else
            continue;
    }
    cout << "Files ready\n";
    sort(files.begin(), files.end());

    //Listing file names:
    // for(size_t i = 1;i<files.size();i++){
    //     cout<<files[i]<<endl;
    // }

    //Core of the program:
    for (size_t i = 1; i < files.size(); i++)
    {

        pcl::io::loadPCDFile(("../../resources/data_1/" + files[i]), *cloud);

        cout << "\n\nCloud loaded, size: " << cloud->points.size() << endl;

        //Maybe check if XYZRGB and turn it into XYZ

        //Starting the process
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        //Crop box filter
        pcl::PointCloud<PointType>::Ptr cropBoxCloud = cropBoxFilter(cloud);
        //Voxel filter
        pcl::PointCloud<PointType>::Ptr voxelCloud = voxelFilter(cropBoxCloud);

        //RANSAC segmentation
        pair<typename pcl::PointCloud<PointType>::Ptr, typename pcl::PointCloud<PointType>::Ptr> planeAndObjects = RANSACsegmentation(voxelCloud);
        pcl::PointCloud<PointType>::Ptr plane = planeAndObjects.first;
        pcl::PointCloud<PointType>::Ptr objects = planeAndObjects.second;
        //Removing Noise from objects
        pcl::PointCloud<PointType>::Ptr noiseFreeobjects = noiseRemoval(objects);
        //Cluster Cars
        vector<pcl::PointCloud<PointType>::Ptr> clusters = extractClusters(noiseFreeobjects);

        //Visualization
        cout << "\nVisualizing frame " << frame << " ...\n";
        Eigen::Vector3f color0(0, 0, 0);
        // pcl::visualization::PCLVisualizer::Ptr viewer1 = viewPointCloud(cloud, "Original Point Cloud");
        // pcl::visualization::PCLVisualizer::Ptr viewer2 = viewPointCloud(voxelCloud, "Voxel Filter");
        // pcl::visualization::PCLVisualizer::Ptr viewer3 = viewPointCloud(cropBoxCloud, "Crop Box Filter",color0);
        // pcl::visualization::PCLVisualizer::Ptr viewer4 = viewPointCloud(plane, "Plane",color0);
        // pcl::visualization::PCLVisualizer::Ptr viewer5 = viewPointCloud(objects, "Objects",color0);

        viewer->setBackgroundColor(0, 0, 0);
        //Create bounding rectangles for clusters
        createBoundingBoxes(clusters, viewer);
        //Set camera position
        viewer->setCameraPosition(-30, 0, 10, 0, 0, 0, 0, 0, 1);

        Eigen::Vector3f color1(0, 0, 1);
        Eigen::Vector3f color2(0, 1, 0);

        addPointCloudToViewer(viewer, noiseFreeobjects, "Objects", color1);
        addPointCloudToViewer(viewer, plane, "road", color2);

        // while(viewer->wasStopped()){
        //     viewer->spinOnce(1);
        // }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        cout << "Processing duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        //Earch frame will last for this amount of ms
        viewer->spinOnce(4000);

        // Clear the viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        //Increment the frame number
        frame++;

    } //end main loop

    cout << "\nDone visualizing\n\n";

    cout << "Program ended" << endl;
    return 0;
}
