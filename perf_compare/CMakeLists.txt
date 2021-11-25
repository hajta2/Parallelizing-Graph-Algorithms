set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable(run_csr perf_compare.cpp)

target_link_libraries(
    run_csr
    OpenMP::OpenMP_CXX
    MMIO
    CLI11
    project_options
    project_warnings
    Boost::boost)