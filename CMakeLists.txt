cmake_minimum_required(VERSION 3.18)
project(Parallelizing_Graph_Algorithms)
set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
else()
    # GCC
    set(MKLROOT
        "$ENV{MKLROOT}"
        CACHE INTERNAL "Copied from environment variable")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS}  -m64  -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
    )
endif()
add_subdirectory(lib/MMIO)
file(GLOB SOURCES "src/*.cpp")
add_executable(Parallelizing_Graph_Algorithms ${SOURCES})
target_link_libraries(Parallelizing_Graph_Algorithms OpenMP::OpenMP_CXX MMIO)
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin/)

include_directories(include lib)
