cmake_minimum_required(VERSION 3.12)
project(bmf_caption_modules LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -fopenmp -fPIC -DBYTED_FFMPEG_OPT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s -O3")
set(CMAKE_DEBUG_POSTFIX _d)
if(DEFINED ENV{BUILD_VERSION})
        message(STATUS "VERSION is $ENV{BUILD_VERSION}")
        set(BM_BUILD_VERSION $ENV{BUILD_VERSION})
else()
        message(STATUS "Cannot find version info. using default value 1.0.0.0")
        set(BM_BUILD_VERSION "1.0.0.0")
endif()

# CUDA
set(CMAKE_CUDA_STANDARD 14)
set(CUDA_ARCH "70;75")
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-declarations")
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas --warn-on-spills")
endif ()
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 10.2.89)
        message(STATUS "cuda version: " ${CMAKE_CUDA_COMPILER_VERSION} ". add Ampere support.")
        set(CUDA_ARCH "70;75;80;86")
endif()
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

if(NOT DEFINED TRT_LLM_ROOT)
        set(TRT_LLM_ROOT ${PROJECT_SOURCE_DIR}/../../../3rd_party/tensorrt_llm)
endif()
if(NOT DEFINED LIBTORCH_ROOT)
        set(LIBTORCH_ROOT ${PROJECT_SOURCE_DIR}/../../../3rd_party/libtorch)
endif()
if(NOT DEFINED OPENCV_ROOT)
        set(OPENCV_ROOT ${PROJECT_SOURCE_DIR}/../../../3rd_party/opencv)
endif()
if(NOT DEFINED BMF_ROOT)
        set(BMF_ROOT ${PROJECT_SOURCE_DIR}/../../../build/output/bmf)
endif()

set(TRT_LLM_LIBRARY_DIR ${TRT_LLM_ROOT}/lib)
set(TRT_LLM_INCLUDE_DIR ${TRT_LLM_ROOT}/include)
set(LIBTORCH_LIBRARY_DIR ${LIBTORCH_ROOT}/lib)
set(LIBTORCH_INCLUDE_DIR ${LIBTORCH_ROOT}/include)
set(OPENCV_LIBRARY_DIR ${OPENCV_ROOT}/lib)
set(OPENCV_INCLUDE_DIR ${OPENCV_ROOT}/include/opencv4)
set(BMF_LIBRARY_DIR ${BMF_ROOT}/lib)
set(BMF_INCLUDE_DIR ${BMF_ROOT}/include)

add_subdirectory(module)
add_subdirectory(tokenizer)
add_subdirectory(gpu)

set(_GLIBCXX_USE_CXX11_ABI 1)
# output dir
# library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/output/lib)
# binary
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/output/bin)

add_library(bcm SHARED
                $<TARGET_OBJECTS:module>
                $<TARGET_OBJECTS:tokenizer>
                $<TARGET_OBJECTS:gpu>
)
set_target_properties(bcm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(bcm PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

target_link_directories(bcm PUBLIC
                        ${BMF_LIBRARY_DIR}
                        ${OPENCV_LIBRARY_DIR}
                        /usr/local/cuda/lib64
                        ${TRT_LLM_LIBRARY_DIR}
                        ${LIBTORCH_LIBRARY_DIR}
)
target_link_libraries(bcm PUBLIC
                bmf_module_sdk hmp glog cudart cudart_static dl rt spdlog
                opencv_core opencv_imgproc opencv_imgcodecs 
                re2 jpeg c10 torch_cpu nvinfer_plugin nvinfer
                avutil postproc avcodec avformat avfilter avdevice swscale swresample
                nvinfer_plugin_tensorrt_llm tensorrt_llm tensorrt_llm_nvrtc_wrapper th_common
                )

add_definitions(-DHAVE_LIBJPEG)
