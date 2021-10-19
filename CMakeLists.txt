cmake_minimum_required(VERSION 3.11)
project(Parallelizing_Graph_Algorithms)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS -fopenmp)

add_library(project_options INTERFACE)

# sanitizer options if supported by compiler
# include(cmake/Sanitizers.cmake)
# enable_sanitizers(project_options)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

add_subdirectory(lib/MMIO)
file(GLOB SOURCES "src/*.cpp")
add_executable(Parallelizing_Graph_Algorithms ${SOURCES})

target_link_libraries(
    Parallelizing_Graph_Algorithms
    OpenMP::OpenMP_CXX
    MMIO
    project_options
    project_warnings)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/)

include_directories(include)
include_directories(SYSTEM lib)
