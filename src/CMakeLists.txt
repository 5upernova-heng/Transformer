cmake_minimum_required(VERSION 3.16)
project(Transformer)

add_subdirectory(matrix)
add_subdirectory(module)

set(SOURCE_FILES main.cpp)

add_executable(Transformer ${SOURCE_FILES})
target_link_libraries(Transformer matrix module)
