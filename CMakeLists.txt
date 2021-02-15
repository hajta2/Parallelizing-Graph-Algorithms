cmake_minimum_required(VERSION 3.17)
project(Parallelizing_Graph_Algorithms)

set(CMAKE_CXX_STANDARD 20)

add_executable(Parallelizing_Graph_Algorithms main.cpp graphDense.hpp graphCompressed.hpp graphCoordinate.hpp abstractGraph.hpp)