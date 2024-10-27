set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(UNIX)
    # hidden symbols by default & static library compilation
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang") # IOS -> AppleClang
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android") # FIXME
            add_link_options(-undefined error)
        endif()
    else()
        add_link_options(-Wl,--no-undefined)
    endif()
endif()

set(HMP_CORE_PUB_DEPS)
set(HMP_CORE_PRI_DEPS)

set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

#### pybind11
if(HMP_ENABLE_PYTHON)
    if (HMP_LOCAL_DEPENDENCIES)
        add_subdirectory(third_party/pybind11)
    else ()
        find_package(pybind11 REQUIRED)
    endif()
endif()

##### fmt
if (HMP_LOCAL_DEPENDENCIES)
    add_subdirectory(third_party/fmt)
else ()
    find_package(fmt REQUIRED)
    add_library(fmt ALIAS fmt::fmt)
endif()
list(APPEND HMP_CORE_PUB_DEPS fmt)

##### Torch
if(HMP_ENABLE_TORCH)
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

##### optional(remove it when nvcc support c++17)
add_library(optional INTERFACE)   
set_target_properties(optional PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/third_party/optional/include)
list(APPEND HMP_CORE_PUB_DEPS optional)


#### spdlog
if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Android|iOS" AND NOT EMSCRIPTEN)
    if (HMP_LOCAL_DEPENDENCIES)
        add_subdirectory(third_party/spdlog)
        set_target_properties(spdlog PROPERTIES
                C_VISIBILITY_PRESET hidden
                CXX_VISIBILITY_PRESET hidden
        )
        list(APPEND HMP_CORE_PRI_DEPS spdlog)
    else ()
        find_package(spdlog REQUIRED)
        add_library(spdlog ALIAS spdlog::spdlog)
        list(APPEND HMP_CORE_PUB_DEPS spdlog)
    endif()

endif()

#### dlpack
if (NOT HMP_LOCAL_DEPENDENCIES)
    find_package(dlpack REQUIRED)
    list(APPEND HMP_CORE_PRI_DEPS dlpack::dlpack)
endif()

#### backward-cpp
if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Android|iOS")
    if (HMP_LOCAL_DEPENDENCIES)
        find_library(LIBDW dw)
        if("${LIBDW}" MATCHES "LIBDW-NOTFOUND")
            set(STACK_DETAILS_BFD TRUE)
            set(STACK_DETAILS_AUTO_DETECT FALSE)
        endif()

        add_subdirectory(third_party/backward)
        list(APPEND HMP_CORE_PRI_DEPS backward ${BACKWARD_LIBRARIES})
    else ()
        find_package(Backward REQUIRED)
        list(APPEND HMP_CORE_PUB_DEPS Backward::Backward)
    endif()
endif()

##### CUDA
if(HMP_ENABLE_CUDA)
    include(cmake/cuda.cmake)

    if(NOT CUDAToolkit_FOUND)
        message("CUDA not found, disable it!")
        set(HMP_ENABLE_CUDA OFF)
        set(HMP_ENABLE_NPP OFF)
    else()
        list(APPEND HMP_CORE_PUB_DEPS cuda::cuda)
        list(APPEND HMP_CORE_PRI_DEPS cuda::cudart)

        if(HMP_ENABLE_NPP)
            list(APPEND HMP_CORE_PRI_DEPS cuda::npp)
        endif()
    endif()
endif()

##### FFMPEG
if(HMP_ENABLE_FFMPEG)
    include(cmake/ffmpeg.cmake)
    if(NOT FFMPEG_FOUND)
        message("FFMEPG library not found, disable it")
        set(HMP_ENABLE_FFMPEG OFF)
    else()
        list(APPEND HMP_CORE_PRI_DEPS 
            ffmpeg::avutil ffmpeg::avcodec ffmpeg::avformat)
    endif()
endif()

##### libyuv(deprecated)
#add_subdirectory(third_party/libyuv)
#target_include_directories(yuv
#    INTERFACE
#        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/libyuv/include>)
#list(APPEND HMP_CORE_PRI_DEPS yuv)


##### OpenCV
if(HMP_ENABLE_OPENCV)
    include(cmake/opencv.cmake)
    if(OpenCV_FOUND)
        list(APPEND HMP_CORE_PUB_DEPS ${OpenCV_LIBS})
    else()
        message("OpenCV not found, disable it")
        set(HMP_ENABLE_OPENCV OFF)
    endif()
endif()


##### GTest
if (HMP_LOCAL_DEPENDENCIES)
    if(WIN32)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(third_party/gtest)
else ()
    find_package(GTest REQUIRED)
    add_library(gtest ALIAS GTest::gtest)
    add_library(gtest_main ALIAS GTest::gtest_main)
endif()


##### Benchmark
if (HMP_LOCAL_DEPENDENCIES)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_INSTALL OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    if (NOT EMSCRIPTEN)
        add_subdirectory(third_party/benchmark)
    endif()
else ()
    find_package(benchmark REQUIRED)
endif()


##### OpenMP
if(HMP_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message("OpenMP not found, disable it")
        set(HMP_ENABLE_OPENMP OFF)
    endif()
endif()


if(HMP_ENABLE_JNI)
    if(NOT ANDROID)
        find_package(JNI)
        if(NOT JNI_FOUND)
            message("JNI not found, disable it")
            set(HMP_ENABLE_JNI FALSE)
        endif()
    else()
        set(JNI_INCLUDE_DIRS)
        set(JNI_LIBRARIES)
    endif()
endif()

set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_OLD})