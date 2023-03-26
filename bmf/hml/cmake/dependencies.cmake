
if(UNIX)
    # hidden symbols by default & static library compilation
    add_compile_options(-fPIC)
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
    add_subdirectory(third_party/pybind11)
endif()


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
    set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
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


##### fmt
add_subdirectory(third_party/fmt)
list(APPEND HMP_CORE_PRI_DEPS fmt)

##### optional(remove it when nvcc support c++17)
add_library(optional INTERFACE)   
set_target_properties(optional PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/third_party/optional/include)
list(APPEND HMP_CORE_PUB_DEPS optional)


#### spdlog
if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Android|iOS")
    add_subdirectory(third_party/spdlog)
    list(APPEND HMP_CORE_PRI_DEPS spdlog)
    target_compile_options(spdlog PRIVATE 
        -fvisibility=hidden
    )
endif()


#### backward-cpp
if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Android|iOS")
    find_library(LIBDW dw)
    if("${LIBDW}" MATCHES "LIBDW-NOTFOUND")
        set(STACK_DETAILS_BFD TRUE)
        set(STACK_DETAILS_AUTO_DETECT FALSE)
    endif()
    add_subdirectory(third_party/backward)
    list(APPEND HMP_CORE_PRI_DEPS backward ${BACKWARD_LIBRARIES})
endif()

##### CUDA
if(HMP_ENABLE_CUDA)
    include(cmake/cuda.cmake)

    if(NOT CUDA_FOUND)
        message("CUDA not found, disable it!")
        set(HMP_ENABLE_CUDA OFF)
        set(HMP_ENABLE_NPP OFF)
    else()
        list(APPEND HMP_CORE_PRI_DEPS cuda::cuda cuda::cudart)

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
        list(APPEND HMP_CORE_PRI_DEPS ${OpenCV_LIBS})
    else()
        message("OpenCV not found, disable it")
        set(HMP_ENABLE_OPENCV OFF)
    endif()
endif()


##### GTest
add_subdirectory(third_party/gtest)


##### Benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
add_subdirectory(third_party/benchmark)


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