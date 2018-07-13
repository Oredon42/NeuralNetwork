cmake_minimum_required(VERSION 3.0)

project(Test)

set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

# Build flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 /Od /Zi /Wall")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-std=c++11 /O2 /fp:fast /arch:SSE2 /Gv /Oi /Zi /Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 /O2 /fp:fast /arch:SSE2 /Gv /Oi /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-std=c++11 -O2 -ffast-math -mfpmath=sse -msse2 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O2 -ffast-math -mfpmath=sse -msse2 -DNDEBUG")
endif()

# Sources
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SOURCE_FILES
    src/main.cpp
)
set(HEADER_FILES
    src/datasetgenerator.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

if(MSVC)
    source_group(TREE ${SOURCE_DIR} PREFIX "Source Files" FILES ${SOURCE_FILES})
    source_group(TREE ${SOURCE_DIR} PREFIX "Header Files" FILES ${HEADER_FILES})
endif(MSVC)

add_subdirectory(../neural ${CMAKE_BINARY_DIR}/neural)

# Link
target_link_libraries(${PROJECT_NAME}
    NeuralLib
)

target_include_directories(${PROJECT_NAME} PRIVATE ${SOURCE_DIR})
