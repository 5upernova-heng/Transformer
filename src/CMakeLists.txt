cmake_minimum_required(VERSION 3.16)
project(Transformer)

add_subdirectory(matrix)
add_subdirectory(module)

find_package(Torch REQUIRED)
find_package(BLAS REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES main.cpp
        utils.cpp
        utils.h
        main.h)

add_executable(Transformer ${SOURCE_FILES})
target_link_libraries(Transformer module matrix "${TORCH_LIBRARIES}")
