cmake_minimum_required(VERSION 3.5)
project(gp3_from_centered)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(gp3_from_centered gp3_from_centered.cpp)
target_link_libraries(gp3_from_centered ${PCL_LIBRARIES})
