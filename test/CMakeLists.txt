cmake_minimum_required(VERSION 3.2)
project(tests)

add_subdirectory(lib/googletest)

include_directories(${MATRIX_HEADERS_DIR})
include_directories(${MODULE_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

set(SOURCE_FILES main.cpp src/matrix_test.cpp)

add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests matrix module gtest)
