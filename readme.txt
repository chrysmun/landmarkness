Welcome!

This is an implementation of the Kobyshev score from the 3DV 2016 paper.
This is only the initial score, the refined score code is pending...

3D Saliency for Finding Landmark Buildings, 3DV 2016
N. Kobyshev, H. Riemenschneider, A. Bodis-Szomoru, L. Van Gool
http://varcity.eu/publication.html

/// requirements
* PCL v1.7
* OMP [optional]

/// compile.sh

mkdir build
cd build
cmake ..
make -j8

/// run.sh
build/landmarkness ../data/fraumunster.ply






/// example output:
  Reading ../data/fraumunster.ply
  Loaded 405575 points.
  Execution time: 0.0643783
  computing normals... 
  done recting 405575 normal directions
  Execution time: 0.50857
  computing FPFH... 
  done compusting 405575 feature descriptors
  Execution time: 48.2682
  computing knn neighbors
  Execution time: 2.38485
  done measuring 20278750 neighbors (avg=50)
  landmark min=-3.91202 max=0
  Execution time: 259.573
  computing visualization cloud 
  Execution time: 2.09063
  ka-pow!
