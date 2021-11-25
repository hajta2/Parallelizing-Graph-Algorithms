cmake_minimum_required(VERSION 3.18)
project(Parallelizing_Graph_Algorithms)
set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

option(USE_VCL "Building with VCL support, Intel compilers does NOT support these" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    message(STATUS "Intel compiler detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
    set(USE_VCL
        OFF
        CACHE BOOL "" FORCE)
else()
    # GCC
    set(MKLROOT
        "$ENV{MKLROOT}"
        CACHE INTERNAL "Copied from environment variable")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS}  -m64  -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
    )
endif()

add_library(project_options INTERFACE)

# sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
# include(cmake/CompilerWarnings.cmake)
# set_project_warnings(project_warnings)

add_subdirectory(lib/MMIO)
add_subdirectory(lib/CLI11)
file(GLOB SOURCES "src/*.cpp")
add_executable(Parallelizing_Graph_Algorithms ${SOURCES})
if(USE_VCL)
  target_compile_definitions(project_options INTERFACE USE_VCL_LIB)
endif(USE_VCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(Boost REQUIRED)

target_include_directories(project_options INTERFACE include)
target_include_directories(project_options SYSTEM INTERFACE lib)

target_link_libraries(
    Parallelizing_Graph_Algorithms
    OpenMP::OpenMP_CXX
    MMIO
    CLI11
    project_options
    project_warnings
    Boost::boost)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/)

include_directories(include)
include_directories(SYSTEM lib)
