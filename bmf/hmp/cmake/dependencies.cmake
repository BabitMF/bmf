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

set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/custom_deps)

include(FetchContent)


#### pybind11
if(HMP_ENABLE_PYTHON)
    if (HMP_LOCAL_DEPENDENCIES)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG 2e0815278cb899b20870a67ca8205996ef47e70f
        )
        FetchContent_MakeAvailable(pybind11)
    else ()
        find_package(pybind11 REQUIRED)
    endif()
endif()

##### fmt
if (HMP_LOCAL_DEPENDENCIES)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 6ae402fd0bf4e6491dc7b228401d531057dbb094
    )
    FetchContent_MakeAvailable(fmt)
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

#### spdlog
if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Android|iOS" AND NOT EMSCRIPTEN)
    if (HMP_LOCAL_DEPENDENCIES)
        FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG be14e60d9e8be31735dd9d2d132d8a4cd3482165
        )
        FetchContent_MakeAvailable(spdlog)
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
if (HMP_LOCAL_DEPENDENCIES)
    FetchContent_Declare(
        dlpack
        GIT_REPOSITORY https://github.com/dmlc/dlpack.git
        GIT_TAG ca4d00ad3e2e0f410eeab3264d21b8a39397f362
    )
    FetchContent_MakeAvailable(dlpack)
else ()
    find_package(dlpack REQUIRED)
    add_library(dlpack ALIAS dlpack::dlpack)
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

        FetchContent_Declare(
            backward
            GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
            GIT_TAG 872350775655ad610f66aea325c319950daa7c95
        )
        FetchContent_MakeAvailable(backward)

        list(APPEND HMP_CORE_PRI_DEPS backward ${BACKWARD_LIBRARIES}) # assume BACKWARD_LIBRARIES is populated by FetchContent
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


##### Testing Framework (FuzzTest or GTest)
if(BMF_ENABLE_FUZZTEST AND HMP_LOCAL_DEPENDENCIES) # FuzzTest
    # The optional interface library is omitted when using FuzzTest because it conflicts with the optional target defined by fuzztest/abseil-cpp 
    # TODO: Add proper namespacing of fuzztest and associated dependencies to avoid target name conflicts
    if(WIN32)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        fuzztest
        GIT_REPOSITORY https://github.com/google/fuzztest
        GIT_TAG 9f67235e7933e0f626f16977855ec99c0f64f4e0
    )
    FetchContent_MakeAvailable(fuzztest)
else() # GTest
    # optional interface library (remove it when nvcc support c++17)
    add_library(optional INTERFACE)
    set_target_properties(optional PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/third_party/optional/include)
    list(APPEND HMP_CORE_PUB_DEPS optional)

    if(HMP_LOCAL_DEPENDENCIES)
        if(WIN32)
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        endif()
        FetchContent_Declare(
            gtest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG 3ff1e8b98a3d1d3abc24a5bacb7651c9b32faedd
        )
        FetchContent_MakeAvailable(gtest)
    else()
        find_package(GTest REQUIRED)
        add_library(gtest ALIAS GTest::gtest)
        add_library(gtest_main ALIAS GTest::gtest_main)
    endif()
endif()


##### Benchmark
if (HMP_LOCAL_DEPENDENCIES)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_INSTALL OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    if (NOT EMSCRIPTEN)
        FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.0
        )
        FetchContent_MakeAvailable(benchmark) 
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
