cmake_minimum_required(VERSION 3.16)
project(module C CXX)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES
        attention.h
        attention.cpp
        feedforward.cpp
        feedforward.h
)

add_library(module SHARED STATIC ${SOURCE_FILES})
