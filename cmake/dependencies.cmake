

# python

if(BMF_ENABLE_PYTHON)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        find_package(Python ${BMF_PYENV} EXACT REQUIRED COMPONENTS Development) # python dist
    else()
        find_package(Python ${BMF_PYENV} COMPONENTS Interpreter Development) # python dist
    endif()
endif()

### json
add_library(nlohmann INTERFACE IMPORTED GLOBAL)
set_property(TARGET nlohmann PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}/3rd_party/json/include)

## glog
if(BMF_EANBLE_GLOG)
    include(cmake/glog.cmake)
endif()

## gtest
# use target from hml

## ffmepg
if(BMF_ENABLE_FFMPEG)
    include(cmake/ffmpeg.cmake)
    if(FFMPEG_FOUND)
        set(BMF_FFMPEG_TARGETS 
            ffmpeg::avcodec ffmpeg::avformat ffmpeg::avfilter 
            ffmpeg::avdevice ffmpeg::avutil ffmpeg::swscale ffmpeg::swresample)
    else()
        set(BMF_FFMPEG_TARGETS)
        message(WARNING "FFMPEG libraries not found, disable it")
        set(BMF_ENABLE_FFMPEG FALSE)
    endif()
endif()

## CUDA
if(BMF_ENABLE_CUDA)
    set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
    find_package(CUDA QUIET)
    if(NOT CUDA_FOUND)
        set(BMF_ENABLE_CUDA OFF)
    endif()
endif()

# JNI
if(BMF_ENABLE_JNI)
    if(NOT ANDROID)
        find_package(JNI)
        if(NOT JNI_FOUND)
            message("JNI not found, disable it")
            set(BMF_ENABLE_JNI FALSE)
        endif()
    else()
        set(JNI_INCLUDE_DIRS)
        set(JNI_LIBRARIES)
    endif()
endif()

## HML
# disable torch build, as -D_GLIBCXX_USE_CXX11_ABI=0 will make gtest build failed
set(HMP_ENABLE_TORCH OFF) 
set(HMP_ENABLE_FFMPEG OFF)  # remove ffmpeg dependencies
set(HMP_ENABLE_OPENCV OFF)  # remove opencv dependencies
set(HMP_ENABLE_OPENMP OFF)  # remove openmp dependencies
set(HMP_ENABLE_CUDA ${BMF_ENABLE_CUDA})
set(HMP_ENABLE_PYTHON ${BMF_ENABLE_PYTHON})
set(HMP_ENABLE_JNI ${BMF_ENABLE_JNI})
set(HMP_ENABLE_MOBILE ${BMF_ENABLE_MOBILE})
add_subdirectory(bmf/hml)

## cuda driver api
if(BMF_ENABLE_CUDA)
    find_package(CUDA QUIET)
    find_library(CUDA_LIB cuda 
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs lib/x64)
    add_library(cuda::cuda UNKNOWN IMPORTED)   
    set_target_properties(cuda::cuda PROPERTIES
        IMPORTED_LOCATION ${CUDA_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})
endif()


## standard deps
set(BMF_STD_DEPS dl)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND BMF_STD_DEPS stdc++fs pthread)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang") # AppleClang
    if(ANDROID)
        list(APPEND BMF_STD_DEPS ${ANDROID_STL} log)
    else()
        #list(APPEND BMF_STD_DEPS c++fs pthread)
    endif()
endif()