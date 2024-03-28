cmake_minimum_required(VERSION 3.16)

project(Transformer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(TRANSFORMER_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
# submodules
set(MATRIX_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/matrix)
set(MODULE_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/module)

include_directories(${TRANSFORMER_INSTALL_INCLUDE_DIR})
include_directories(${MATRIX_HEADERS_DIR})
include_directories(${MODULE_HEADERS_DIR})

add_subdirectory(src)
add_subdirectory(test)


