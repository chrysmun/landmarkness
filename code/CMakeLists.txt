cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(landmarkness)


#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_VERBOSE_MAKEFILE on)
SET(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_FLAGS "-DPCL_ONLY_CORE_POINT_TYPES=ON -DNO_EXPLICIT_INSTANTIATIONS") 

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


#SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O3")
#SET(CMAKE_C_FLAGS_DISTRIBUTION "-O3")
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-Ofast")
SET(CMAKE_C_FLAGS_DISTRIBUTION "-Ofast")

# needed for chrono
ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    # Other flags
)

SET(BASE /scratch_net/quoellfrisch/rhayko/student/lib/)

# set share directory to load via find_package
SET(PCL_DIR             ${BASE}/pcl-pcl-1.7.1/local/share/pcl-1.7)
SET(OpenCV_DIR 	        ${BASE}/opencv-2.4.9/local/share/OpenCV)

SET(FLANN_INCLUDE_DIRS  ${BASE}/flann-1.8.4-src/local/include/)
SET(FLANN_LIBRARY       ${BASE}/flann-1.8.4-src/local/lib/libflann_cpp_s.a)
SET(EIGEN_INCLUDE_DIRS  ${BASE}/eigen-3.1.2/local/include/eigen3)

# let findpackage find the correct paths
find_package(PCL 1.7 REQUIRED)
#find_package(OpenCV 2.4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${FLANN_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${FLANN_LIBRARY})
add_definitions(${PCL_DEFINITIONS})

add_executable (landmarkness landmarkness.cpp)
target_link_libraries (landmarkness ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})


