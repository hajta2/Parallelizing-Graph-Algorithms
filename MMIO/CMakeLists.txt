add_library(MMIO SHARED mmio.cpp)

target_include_directories(MMIO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
