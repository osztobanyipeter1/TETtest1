cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(surface_reconstruction)

find_package(PCL 1.12 REQUIRED COMPONENTS common io search kdtree features surface visualization filters)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(surface_reconstruction surface_reconstruction.cpp)
target_link_libraries(surface_reconstruction ${PCL_LIBRARIES})