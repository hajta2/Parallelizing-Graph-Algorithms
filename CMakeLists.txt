cmake_minimum_required(VERSION 3.18)
project(Parallelizing_Graph_Algorithms)

set(CMAKE_CXX_STANDARD 20)
if(NOT
   CMAKE_CXX_COMPILER_ID
   MATCHES
   "Intel")
    message(FATAL_ERROR "Only Intel compilers supported!")
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")

add_subdirectory(MMIO)

add_executable(Parallelizing_Graph_Algorithms main.cpp)


target_link_libraries(Parallelizing_Graph_Algorithms OpenMP::OpenMP_CXX MMIO)
