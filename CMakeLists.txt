cmake_minimum_required(VERSION 3.11)
project(Parallelizing_Graph_Algorithms)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

#gcc with armpl
set(ARMPL_INCLUDES
    "$ENV{ARMPL_INCLUDES}"
    CACHE INTERNAL "Copied from environment variable")
set(ARMPL_DIR
    "$ENV{ARMPL_DIR}"
    CACHE INTERNAL "Copied from environment variable")

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -I$ARMPL_INCLUDES -L$ARMPL_DIR/lib -larmpl_mp -lamath -lm"
)
#set(CMAKE_CXX_FLAGS
#    "${CMAKE_CXX_FLAGS} -march=native"
#)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -march=armv8.6-a+sve2"
)

#add_library(project_options INTERFACE)

# sanitizer options if supported by compiler
# include(cmake/Sanitizers.cmake)
# enable_sanitizers(project_options)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# add_subdirectory(lib/MMIO)
add_subdirectory(lib/CLI11)
file(GLOB SOURCES "src/*.cpp")
add_executable(Parallelizing_Graph_Algorithms ${SOURCES})

target_link_libraries(
    Parallelizing_Graph_Algorithms
    OpenMP::OpenMP_CXX
    Boost::boost
    CLI11
    # MMIO
    #project_options
    project_warnings)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/)

include_directories(include)
include_directories(SYSTEM lib)