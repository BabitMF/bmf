cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

cmake_policy(SET CMP0077 NEW)

project(HML CXX C)

### general settings
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
    set(_HMP_MAIN_PROJECT TRUE)
else()
    set(_HMP_MAIN_PROJECT FALSE)
endif()


### options
option(HMP_ENABLE_CUDA "Enable CUDA device support" ON)
option(HMP_ENABLE_FFMPEG "Enable FFMEPG support" ON)
option(HMP_ENABLE_OPENCV "Enable OpenCV support" ON)
option(HMP_ENABLE_NPP "Enable NPP support" OFF)
option(HMP_STATIC_LINK_CUDA "Using static cuda libs" ON)
option(HMP_ENABLE_OPENMP "Enable OPENMP support" ON)
option(HMP_ENABLE_TORCH "Enable Troch support" ON)
option(HMP_ENABLE_PYTHON "Enable Python support" ON)
option(HMP_ENABLE_JNI "Enable build with JNI support" ON)
option(HMP_ENABLE_MOBILE "Enable build for mobile device" OFF)
set(HMP_CUDA_ARCH_FLAGS 7.0 7.5)


### version
set(HMP_VERSION_MAJOR 1)
set(HMP_VERSION_MINOR 2)
set(HMP_VERSION_PATCH 0)
set(HMP_BUILD_VERSION "${HMP_VERSION_MAJOR}.${HMP_VERSION_MINOR}.${HMP_VERSION_PATCH}"
    CACHE STRING "HMP default build version")
if(DEFINED ENV{HMP_BUILD_VERSION})
    set(HMP_BUILD_VERSION "$ENV{HMP_BUILD_VERSION}" 
        CACHE STRING "HMP build version" FORCE)
endif()


### dependencies
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS TRUE)
endif()

include(cmake/dependencies.cmake)

## 
enable_testing()

### 
configure_file(
    "${PROJECT_SOURCE_DIR}/include/hmp/config.h.in"
    "${PROJECT_SOURCE_DIR}/include/hmp/config.h")
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/hml-config.cmake.in"
    "${PROJECT_SOURCE_DIR}/cmake/hml-config.cmake" @ONLY)

add_subdirectory(src)
if(HMP_ENABLE_PYTHON)
    add_subdirectory(py)
endif()
if(HMP_ENABLE_JNI)
    add_subdirectory(java)
endif()

if(NOT HMP_ENABLE_MOBILE)
    add_subdirectory(perf)
endif()

if(IOS)
    add_subdirectory(object-c)
endif()

## generate setup.py
configure_file(
    "${PROJECT_SOURCE_DIR}/setup.py.in"
    "${PROJECT_SOURCE_DIR}/setup.py")

### print build configurations
include(cmake/summary.cmake)