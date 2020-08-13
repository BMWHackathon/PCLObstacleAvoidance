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

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

typedef pcl::PointXYZ PointType;

using namespace std;

//  Parameters

float voxelLeafSize;                                        //0.2
float point1x, point1y, point1z, point2x, point2y, point2z; //-11 -6 -11    19 7 19
Eigen::Vector4f cropBoxMinPoint;
Eigen::Vector4f cropBoxMaxPoint;

//

pcl::visualization::PCLVisualizer::Ptr viewPointCloud(pcl::PointCloud<PointType>::Ptr cloud, string title)
{

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0, 0, 0);
    
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    //viewer->addCoordinateSystem (1.0, "global");
    //viewer->initCameraParameters();
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
    //RANSAC segmentation
    pair<typename pcl::PointCloud<PointType>::Ptr, typename pcl::PointCloud<PointType>::Ptr> planeAndObjects = RANSACsegmentation(cropBoxCloud);
    pcl::PointCloud<PointType>::Ptr plane = planeAndObjects.first;
    pcl::PointCloud<PointType>::Ptr objects = planeAndObjects.second;
    //Removing Noise from objects
    pcl::PointCloud<PointType>::Ptr noiseFreeobjects = noiseRemoval(objects);
    //Cluster Cars
    vector<pcl::PointCloud<PointType>::Ptr> clusters = extractClusters(noiseFreeobjects);

    //Visualization
    cout << "\nVisualizing ...\n";
    // pcl::visualization::PCLVisualizer::Ptr viewer = viewPointCloud(cloud, "Original Point Cloud");
    // pcl::visualization::PCLVisualizer::Ptr viewer2 = viewPointCloud(voxelCloud, "Voxel Filter");
    pcl::visualization::PCLVisualizer::Ptr viewer3 = viewPointCloud(cropBoxCloud, "Crop Box Filter");
    pcl::visualization::PCLVisualizer::Ptr viewer4 = viewPointCloud(plane, "Plane");
    pcl::visualization::PCLVisualizer::Ptr viewer5 = viewPointCloud(objects, "Objects");
    pcl::visualization::PCLVisualizer::Ptr viewer6 = viewPointCloud(noiseFreeobjects, "Noise free Objects");

    //Create bounding rectangles for clusters
    createBoundingBoxes(clusters, viewer6);

    while (!viewer3->wasStopped())
    {
        viewer3->spinOnce();
    }
    cout << "\nDone visualizing\n\n";

    cout << "Program ended" << endl;
    return 0;
}
