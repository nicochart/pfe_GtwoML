cmake_minimum_required(VERSION 3.18.4...3.21.1)

project(GtwoML)

add_subdirectory(PageRank)
add_subdirectory(BrainMatrixGenerator)

# MPI
find_package(MPI REQUIRED)
