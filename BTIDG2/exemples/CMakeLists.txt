add_executable(BTIG2_A BTIG2_adjacency.c)
add_executable(BTIG2_AT BTIG2_adjacency_transposed.c)

#Maths
target_link_libraries(BTIG2_A PRIVATE m)
target_link_libraries(BTIG2_AT PRIVATE m)

# MPI
find_package(MPI REQUIRED)
target_link_libraries(BTIG2_A PUBLIC MPI::MPI_CXX)
target_link_libraries(BTIG2_AT PUBLIC MPI::MPI_CXX)
