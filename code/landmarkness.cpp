//  computes the kobyshev score for a 3d pointcloud (e.g. landmarkness)
// 
//  3D Saliency for Finding Landmark Buildings, 3DV 2016
//  N. Kobyshev, H. Riemenschneider, A. Bodis-Szomoru, L. Van Gool
//  http://varcity.eu/publication.html
//
//  Copyright (C) 2016 hayko riemenschneider
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream> 
#include <vector> 

#include <pcl/io/pcd_io.h> 
#include <pcl/io/ply_io.h>

#include <pcl/point_types.h> 
#include <pcl/features/spin_image.h> 
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/common/geometry.h>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <ctime>
#include <chrono>
#include <time.h>
#include <numeric>

using namespace std;
using namespace std::chrono;
                
main (int argc, char** argv) 
{
  // parameters
  float normal_support = 0.5f;
  float fpfh_support = 3.0f;
  #define omp 1;
  int kdtree_knn = 50;
  float landmark_sigma = 10.0f;
  
  
  // example high resolution time
  chrono::high_resolution_clock::time_point time_start = chrono::high_resolution_clock::now();
  chrono::high_resolution_clock::time_point time_stop = chrono::high_resolution_clock::now();
  chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(time_stop - time_start);
  std::cout << "Execution time: " << time_span.count() << std::endl;


  // ---------------------------------------------------------------------
  // read input
  std::string fileName = argv[1]; 
  std::cout << "Reading " << fileName << std::endl; 

  time_start = chrono::high_resolution_clock::now();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPLYFile(fileName, *cloud) == -1)
  { 
    PCL_ERROR ("Couldn't read file"); 
    return (-1); 
  } 

  std::cout << "Loaded " << cloud->points.size() << " points." << std::endl; 
  cout << flush;
  
  time_stop = chrono::high_resolution_clock::now();
  time_span = chrono::duration_cast<chrono::duration<double>>(time_stop - time_start);
  std::cout << "Execution time: " << time_span.count() << std::endl;

  
  // ---------------------------------------------------------------------
  // compute the normals
  std::cout << "computing normals... " << std::endl; 
  time_start = chrono::high_resolution_clock::now();
  #ifdef omp
   pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
  #else
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
  #endif
 
  normalEstimation.setInputCloud (cloud); 

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>); 
  normalEstimation.setSearchMethod (tree); 
  pcl::PointCloud<pcl::Normal>::Ptr cloudWithNormals (new pcl::PointCloud<pcl::Normal>); 
   
  normalEstimation.setRadiusSearch (normal_support); 
  normalEstimation.compute (*cloudWithNormals); 
  std::cout << "done recting " << cloud->points.size() << " normal directions" << std::endl;
  time_stop = chrono::high_resolution_clock::now();
  time_span = chrono::duration_cast<chrono::duration<double>>(time_stop - time_start);
  std::cout << "Execution time: " << time_span.count() << std::endl;


  
  // ---------------------------------------------------------------------
  // Create the FPFH estimation class, and pass the input dataset+normals to it
  // http://pointclouds.org/documentation/tutorials/fpfh_estimation.php
    std::cout << "computing FPFH... " << std::endl;
  time_start = chrono::high_resolution_clock::now();

    // CHOOSE OMP OR NORMAL
  #ifdef omp
   pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
   fpfh.setNumberOfThreads(8);
 #else
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  #endif

  fpfh.setInputCloud (cloud);
  // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);
  fpfh.setInputNormals (cloudWithNormals);

  // Create an empty kdtree representation, and pass it to the FPFH estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>); 
  fpfh.setSearchMethod (tree);

  // Output datasets
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  fpfh.setRadiusSearch (fpfh_support);

  // Compute the features
  fpfh.compute (*fpfhs);
  std::cout << "done compusting " << fpfhs->points.size() << " feature descriptors" << std::endl;
  time_stop = chrono::high_resolution_clock::now();
  time_span = chrono::duration_cast<chrono::duration<double>>(time_stop - time_start);
  std::cout << "Execution time: " << time_span.count() << std::endl;
  

  // debug each fpfh descriptor
  if(0)
  for (int x = 0; x < fpfhs->points.size (); x++)
  {
    for (int i = 0; i < 33; i++)
      std::cout <<  fpfhs->points[x].histogram[i] << " ";
    std::cout << std::endl;
  }
  
  
  // ---------------------------------------------------------------------
  // set up kdtree search for these points

  cout << "computing knn neighbors" << endl;
  time_start = chrono::high_resolution_clock::now();

  
  std::vector<float> pointLandmark(fpfhs->points.size(), 0.0f);
  pcl::KdTreeFLANN<pcl::FPFHSignature33> kdtree;
  kdtree.setInputCloud (fpfhs);
  time_stop = chrono::high_resolution_clock::now();
  time_span = chrono::duration_cast<chrono::duration<double>>(time_stop - time_start);
  std::cout << "Execution time: " << time_span.count() << std::endl;

  time_start = chrono::high_resolution_clock::now();
  std::vector<int> pointIdxNKNSearch(kdtree_knn);
  std::vector<float> pointNKNSquaredDistance(kdtree_knn);
  int kdtree_knn_total = 0;
  float landmark_max = 0;
  float landmark_min = 10^6;


  //#pragma omp parallel for
  //for (size_t fpfh_idx = 0;  fpfh_idx< 20;  fpfh_idx++)
  //  std::cout << fpfh_idx << std::endl;
  

  //#pragma omp parallel for
  for (size_t fpfh_idx = 0;  fpfh_idx< fpfhs->points.size();  fpfh_idx++)
  {
    // select instance to compare to
    pcl::FPFHSignature33 searchPoint = fpfhs->points[fpfh_idx];
    float feature_spread = 0.0f;
    float spatial_spread = 0.0f;
    
    if ( kdtree.nearestKSearch (searchPoint, kdtree_knn, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
    {
      for (size_t knn_idx = 0; knn_idx < pointIdxNKNSearch.size(); knn_idx++)
      {

        // initial landmark saliency scored over knn neighbors
        // equation 1
        //pointLandmark[fpfh_idx] += exp(-pointNKNSquaredDistance[knn_idx] / 2 / landmark_sigma);


        // todo: not working yet
        //  float l2 = pcl::geometry::distance (cloud->points[pointIdxNKNSearch[knn_idx]],cloud->points[fpfh_idx]);
        

         float l2 = (
        (cloud->points[pointIdxNKNSearch[knn_idx]].x-cloud->points[fpfh_idx].x)*(cloud->points[pointIdxNKNSearch[knn_idx]].x-cloud->points[fpfh_idx].x)
        + (cloud->points[pointIdxNKNSearch[knn_idx]].y-cloud->points[fpfh_idx].y)*(cloud->points[pointIdxNKNSearch[knn_idx]].y-cloud->points[fpfh_idx].y)
        + (cloud->points[pointIdxNKNSearch[knn_idx]].z-cloud->points[fpfh_idx].z)*(cloud->points[pointIdxNKNSearch[knn_idx]].z-cloud->points[fpfh_idx].z));
        //l2 = sqrt(l2);

        
        feature_spread+=sqrt(pointNKNSquaredDistance[knn_idx]);
        spatial_spread+=sqrt(l2);
                
        pointLandmark[fpfh_idx] += exp(-l2  / 2 / landmark_sigma / landmark_sigma);
        

        // debug each landmark score
        if(0)
        if(knn_idx < 10)
        std::cout << fpfh_idx << ": id="  <<   pointIdxNKNSearch[knn_idx] << "l2=" << l2 << " squared distance=" << pointNKNSquaredDistance[knn_idx] << " uniqueness=" << exp(-l2  / 2 / landmark_sigma) << std::endl;
        
      }
      // scored over knn neighbors and normalized
      // equation 2

      // check for 0 neighbors
      if(pointIdxNKNSearch.size()>0)
      {
        pointLandmark[fpfh_idx] = log(pointLandmark[fpfh_idx] / pointIdxNKNSearch.size());
        
        feature_spread /= pointIdxNKNSearch.size();
        spatial_spread /= pointIdxNKNSearch.size();
        }
      else
        pointLandmark[fpfh_idx] = 0;

      
        kdtree_knn_total +=pointIdxNKNSearch.size();

        if(landmark_max < pointLandmark[fpfh_idx])
          landmark_max = pointLandmark[fpfh_idx] ;

        if(landmark_min > pointLandmark[fpfh_idx])
          landmark_min = pointLandmark[fpfh_idx] ;

          if(0)
          cout << spatial_spread << " " << feature_spread << endl;
    }

  }
  std::cout << "done measuring " << kdtree_knn_total << " neighbors (avg=" << (kdtree_knn_total / fpfhs->points.size()) << ")" << std::endl;
  std::cout << "landmark min=" << landmark_min << " max=" << landmark_max << std::endl;
  time_stop = chrono::high_resolution_clock::now();
  time_span = chrono::duration_cast<chrono::duration<double>>(time_stop - time_start);
  std::cout << "Execution time: " << time_span.count() << std::endl;

  
  // ---------------------------------------------------------------------
  // output point cloud with colors
  cout << "computing visualization cloud " << endl;
  time_start = chrono::high_resolution_clock::now();
  pcl::PointCloud < pcl::PointXYZRGBNormal >::Ptr cloud_colored(new pcl::PointCloud < pcl::PointXYZRGBNormal >);

  for (int i = 0; i < cloudWithNormals->size(); i++) {

    pcl::PointXYZRGBNormal point;
    point.x = cloud->points[i].x;
    point.y = cloud->points[i].y;
    point.z = cloud->points[i].z;

    point.normal_x = cloudWithNormals->points[i].normal_x;
    point.normal_y = cloudWithNormals->points[i].normal_y;
    point.normal_z = cloudWithNormals->points[i].normal_z;

    double logistic = 1-pointLandmark[i] / 4.4;

    // -3.91202
    //  -4.38
    
    // COLORING old
    if (logistic < 0.5)     //B to G
    {
            point.r = 0;
            point.g = 255 * (logistic / 0.5 - 1);
            point.b = 255 - point.g;
    } else                  //G to R
    {
            point.r = 255 * (logistic / 0.5 - 1);
            point.g = 255 - point.r;
            point.b = 0;
    }

    // COLORING
    /*
      float landmarky = 1-pointLandmark[i] / 4;
      point.r = 255 * landmarky;
      point.g = 0;
      point.b = 255 * (1-landmarky);
      */

    cloud_colored->points.push_back(point);


  } // for: each point i

  pcl::PLYWriter writer;
  pcl::PLYReader reader;
  writer.write<pcl::PointXYZRGBNormal> (fileName+"_landmarkness.ply", *cloud_colored, false);
  time_stop = chrono::high_resolution_clock::now();
  time_span = chrono::duration_cast<chrono::duration<double>>(time_stop - time_start);
  std::cout << "Execution time: " << time_span.count() << std::endl;

  std::cout << "ka-pow!" << std::endl;

  // exporting the values
  // values to export:
  // std::vector<float> pointLandmark: computed score
  // std::vector<int> pointIdxNKNSearch: feature knn indices
  // std::vector<float> pointNKNSquaredDistance: feature distances

  string fileout_score=fileName+"_score.bin";
  string fileout_knn_ind=fileName+"_knn_ind.bin";
  string fileout_knn_feat_dist=fileName+"_knn_feat_dist.bin";

  ofstream file_score(fileout_score, ios::out | ios::binary);
  file_score.write((char*)&pointLandmark[0], pointLandmark.size()*sizeof(float));
  file_score.close();

  ofstream file_knn_ind(fileout_knn_ind, ios::out | ios::binary);
  file_knn_ind.write((char*)&pointIdxNKNSearch[0], pointIdxNKNSearch.size()*sizeof(int));
  file_knn_ind.close();

  ofstream file_knn_feat_dist(fileout_knn_feat_dist, ios::out | ios::binary);
  file_knn_feat_dist.write((char*)&pointNKNSquaredDistance[0], pointNKNSquaredDistance.size()*sizeof(int));
  file_knn_feat_dist.close();

  return 0; 
} 
