cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(obstacleAvoidance)
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(obstacleAvoidance main.cpp)
target_link_libraries(obstacleAvoidance ${PCL_LIBRARIES})

SET(COMPILE_FLAGS "-std=c++11")
add_definitions(${COMPILE_FLAGS})
