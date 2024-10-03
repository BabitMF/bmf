

# python

if(BMF_ENABLE_PYTHON)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        find_package(Python ${BMF_PYENV} EXACT REQUIRED COMPONENTS Development) # python dist
    else()
        if(BMF_PYENV)
            find_package(Python ${BMF_PYENV} COMPONENTS Interpreter Development REQUIRED EXACT)
        else()
            find_package(Python ${BMF_PYENV} COMPONENTS Interpreter Development REQUIRED) # python dist
        endif()
    endif()
endif()

if (NOT BMF_LOCAL_DEPENDENCIES)
    find_package(pybind11 REQUIRED)
endif()

# breakpad
if(BMF_ENABLE_BREAKPAD)
    include(cmake/breakpad.cmake)
endif()

### json
if (BMF_LOCAL_DEPENDENCIES)
    add_library(nlohmann INTERFACE IMPORTED GLOBAL)
    set_property(TARGET nlohmann PROPERTY INTERFACE_INCLUDE_DIRECTORIES
            ${PROJECT_SOURCE_DIR}/3rd_party/json/include)
else()
    find_package(nlohmann_json REQUIRED)
    add_library(nlohmann ALIAS nlohmann_json::nlohmann_json)
endif()

## glog
if(BMF_EANBLE_GLOG)
    if (BMF_LOCAL_DEPENDENCIES)
        include(cmake/glog.cmake)
    else()
        find_package(glog REQUIRED)
    endif()
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

##### Torch
if(BMF_ENABLE_TORCH)
    if(PYTHON_EXECUTABLE)
        set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
        set(Python_VERSION ${PYTHON_VERSION})
    endif()
    execute_process (
        COMMAND bash -c "${Python_EXECUTABLE} -c 'import site; print(site.getsitepackages()[0])'"
        OUTPUT_VARIABLE SITE_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CUDNN_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib/libcudnn.so.8)
    find_package(Torch HINTS ${SITE_ROOT})
    find_library(TORCH_PYTHON_LIB torch_python HINTS ${SITE_ROOT}/*/lib)
    if(Torch_FOUND AND TORCH_PYTHON_LIB)
        # -D_GLIBCXX_USE_CXX11_ABI=0 cause other library link failure
        # https://stackoverflow.com/questions/62693218/how-to-solve-gtest-and-libtorch-linkage-conflict
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
        add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
        add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})

        list(APPEND HMP_CORE_PRI_DEPS ${TORCH_LIBRARIES})
    else()
        message("Torch library not found, disable it")
        set(HMP_ENABLE_TORCH OFF)
    endif()
endif()

## CUDA
if(BMF_ENABLE_CUDA)
    find_package(CUDAToolkit 11 QUIET COMPONENTS cudart cuda_driver)
    if(NOT CUDAToolkit_FOUND)
        if (DEFINED $ENV{DEVICE})
            if ($ENV{DEVICE} STREQUAL "gpu")
                message(FATAL_ERROR "cuda not found for gpu generation.")
            endif()
        else()
            message(WARNING "cuda not found, disable it.")
            set(BMF_ENABLE_CUDA OFF)
        endif()
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
set(HMP_LOCAL_DEPENDENCIES ${BMF_LOCAL_DEPENDENCIES})
set(HMP_ENABLE_TORCH ${BMF_ENABLE_TORCH}) 
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
    add_library(cuda::cuda ALIAS CUDA::cuda_driver)
endif()


## standard deps
if(WIN32 AND NOT BMF_LOCAL_DEPENDENCIES)
    find_package(dlfcn-win32 REQUIRED)
    add_library(dl ALIAS dlfcn-win32::dl)
endif()
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
