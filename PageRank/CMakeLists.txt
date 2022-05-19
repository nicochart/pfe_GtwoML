add_executable(PR_OnA OnA/PR_DistributedRV_HardBrain.c)
add_executable(PR_OnA_OpenMP OnA/PR_DistributedRV_HardBrain_OpenMP.c)
add_executable(PR_OnAT OnTransposedA/PR_DistributedRV_HardBrain.c)

#Maths
target_link_libraries(PR_OnA PRIVATE m)
target_link_libraries(PR_OnA_OpenMP PRIVATE m)
target_link_libraries(PR_OnAT PRIVATE m)

# MPI
find_package(MPI REQUIRED)
target_link_libraries(PR_OnA PUBLIC MPI::MPI_CXX)
target_link_libraries(PR_OnA_OpenMP PUBLIC MPI::MPI_CXX)
target_link_libraries(PR_OnAT PUBLIC MPI::MPI_CXX)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
        target_link_libraries(PR_OnA_OpenMP PUBLIC OpenMP::OpenMP_CXX)
endif()
