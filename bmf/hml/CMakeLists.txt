cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

cmake_policy(SET CMP0077 NEW)

project(HML CXX C)

### general settings
if(WIN32)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
    set(_HMP_MAIN_PROJECT TRUE)
else()
    set(_HMP_MAIN_PROJECT FALSE)
endif()


### options
option(HMP_LOCAL_DEPENDENCIES "Build dependencies locally" ON)
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

if (HMP_ENABLE_CUDA)
    # For FindCUDAToolkit support
    cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
endif()

# See https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html for allowed values.
# Default: SASS for all supported architectures and PTX for highest architecture
set(HMP_CUDA_ARCH_FLAGS all CACHE STRING "HMP CUDA architectures")


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
if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS FALSE)
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS TRUE)
endif()

include(cmake/dependencies.cmake)

## 
if (NOT EMSCRIPTEN)
    enable_testing()
endif()

if(FUZZTEST_ENABLE_FUZZING_MODE)
    fuzztest_setup_fuzzing_flags()
endif()

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

if(NOT HMP_ENABLE_MOBILE AND NOT WIN32 AND NOT EMSCRIPTEN)
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