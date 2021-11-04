add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(Catch PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                       $<TARGET_PROPERTY:Catch,INTERFACE_INCLUDE_DIRECTORIES>)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable(tests catch_main.cpp tests.cpp)
target_link_libraries(
    tests
    Catch
    OpenMP::OpenMP_CXX
    project_options
    project_warnings
    Boost::boost)

add_test(
    NAME tests
    COMMAND tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gen_mat generate_matrix.cpp)

target_link_libraries(
    gen_mat
    OpenMP::OpenMP_CXX
    MMIO
    CLI11
    project_options
    project_warnings)

add_executable(run_csr perf_compare.cpp)

target_link_libraries(
    run_csr
    OpenMP::OpenMP_CXX
    MMIO
    CLI11
    project_options
    project_warnings
    Boost::boost)
