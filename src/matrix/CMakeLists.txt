cmake_minimum_required(VERSION 3.16)
project(matrix C CXX)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES
        matrix.h
        matrix.cpp
        )

add_library(matrix SHARED STATIC ${SOURCE_FILES})
