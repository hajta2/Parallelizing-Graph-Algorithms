cmake_minimum_required(VERSION 3.17)
project(Parallelizing_Graph_Algorithms)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(
    Parallelizing_Graph_Algorithms
    main.cpp
    graphDense.hpp
    graphCompressed.hpp
    graphCoordinate.hpp
    abstractGraph.hpp)

target_link_libraries(Parallelizing_Graph_Algorithms OpenMP::OpenMP_CXX)
