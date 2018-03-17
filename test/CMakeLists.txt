set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

project(Test)

cmake_minimum_required(VERSION 2.8)

# Get sources and headers
set(SOURCE_DIR src)

file(GLOB_RECURSE SOURCES ${SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${SOURCE_DIR}/*.h)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -ggdb -O0 -Wall")

include_directories(${CMAKE_SOURCE_DIR}/Neural)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} Neural)

target_include_directories(Test PRIVATE SOURCE_DIR)