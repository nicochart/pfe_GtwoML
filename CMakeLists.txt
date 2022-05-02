cmake_minimum_required(VERSION 3.18.4...3.21.1)

project(GtwoML)

add_executable(PageRankParallele_DistributedRV_HardBrain PageRank/Parallele/OnA/PageRankParallele_DistributedRV_HardBrain.c)
add_executable(PageRankParallele_DistributedRV_HardBrain_OpenMP PageRank/Parallele/OnA/PageRankParallele_DistributedRV_HardBrain_OpenMP.c)

#Maths
target_link_libraries(PageRankParallele_DistributedRV_HardBrain PRIVATE m)
target_link_libraries(PageRankParallele_DistributedRV_HardBrain_OpenMP PRIVATE m)

# MPI
find_package(MPI REQUIRED)
target_link_libraries(PageRankParallele_DistributedRV_HardBrain PUBLIC MPI::MPI_CXX)
target_link_libraries(PageRankParallele_DistributedRV_HardBrain_OpenMP PUBLIC MPI::MPI_CXX)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
        target_link_libraries(PageRankParallele_DistributedRV_HardBrain_OpenMP PUBLIC OpenMP::OpenMP_CXX)
endif()
