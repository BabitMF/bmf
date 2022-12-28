cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Cross Compile
# cmake -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake


project(BMF CXX C)

### options
if (NOT APPLE)
    set(BMF_PYENV 3.7 CACHE STRING "Python version, ie. 3.6, 3.7, 3.8")
else ()
    set(BMF_PYENV)
endif ()
set(BMF_ASSEMBLE_ROOT ${CMAKE_BINARY_DIR}/output CACHE STRING "Directory to assymble BMF package")
set(BMF_BUILD_VERSION "" CACHE STRING "BMF version")
set(BMF_BUILD_COMMIT "" CACHE STRING "BMF commit")

option(BMF_ENABLE_BREAKPAD "Enable build with breakpad support" OFF)


option(BMF_ENABLE_CUDA "Enable CUDA support" ON)
option(BMF_ENABLE_PYTHON "Enable build with python support" ON)
option(BMF_ENABLE_GLOG "Enable build with glog support" OFF)
option(BMF_ENABLE_JNI "Enable build with JNI support" ON)
option(BMF_ENABLE_FFMPEG "Enable build with ffmpeg support" ON)
option(BMF_ENABLE_MOBILE "Enable build for mobile platform" OFF)

if(ENABLE_BITCODE)
    set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE "bitcode")
    set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE "YES") 
else()
    set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")
endif()



### VERSION
string(REPLACE "." ";" VERSION_LIST ${BMF_BUILD_VERSION})
list(GET VERSION_LIST 0 BMF_VERSION_MAJOR)
list(GET VERSION_LIST 1 BMF_VERSION_MINOR)
list(GET VERSION_LIST 2 BMF_VERSION_PATCH)


### general settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (${COVERAGE})
    set(CMAKE_COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage -O -g3")
else ()
    set(CMAKE_COVERAGE_FLAGS " ")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if ($ENV{SCRIPT_EXEC_MODE} MATCHES "x86")
        # runpath for bmf libs
        set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -Wl,-z,defs,-rpath,'$ORIGIN',--enable-new-dtags ${CMAKE_COVERAGE_FLAGS}") # For python dist
    else()
        set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -Wl,-z,defs ${CMAKE_COVERAGE_FLAGS}") #report unresolved symbols
    endif()
endif()

if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -g")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Oz") # -O2 + -Oz

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
endif()


#
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/bin)


### dependencies
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS TRUE)  #build shared sdk by default
endif()

include(cmake/dependencies.cmake)

IF(BMF_USE_MEDIACODEC)

add_definitions("-DBMF_USE_MEDIACODEC")

endif(USE_MEDIACODEC)


### 
add_subdirectory(bmf)

## set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGSa} -Wl,--export-dynamic")


### print build configurations
include(cmake/summary.cmake)
