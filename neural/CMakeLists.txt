set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

project(Neural)

cmake_minimum_required(VERSION 2.8)

# Get sources and headers
set(SOURCE_DIR src)

file(GLOB_RECURSE SOURCES ${SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${SOURCE_DIR}/*.h)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -ggdb -O0 -Wall")

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(Neural PRIVATE SOURCE_DIR)