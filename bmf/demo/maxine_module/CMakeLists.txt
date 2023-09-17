cmake_minimum_required(VERSION 3.12)
project(maxine_module)
# language setting
# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

SET(MAXINE_DIR CACHE STRING "Maxine Installation Directory")

if (NOT DEFINED CUDA_ROOT)
    set(CUDA_ROOT /usr/local/cuda)
endif()
set(CUDA_INCLUDE_DIR ${CUDA_ROOT}/include)
set(CUDA_LIBRARY_DIR ${CUDA_ROOT}/lib64)
message(STATUS "detect cuda root: " ${CUDA_INCLUDE_DIR})

set(MAXINE_INCLUDE_DIR ${MAXINE_DIR}/include)
set(MAXINE_LIBRARY_DIR ${MAXINE_DIR}/lib)

find_package(OpenCV 4.2.0 QUIET COMPONENTS imgproc core imgcodecs)

SET(MODULE_SRCS ${PROJECT_SOURCE_DIR}/maxine_module.cpp)

add_library(${PROJECT_NAME} SHARED ${MODULE_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../bmf/include/
                           ${CUDA_INCLUDE_DIR}
                           ${MAXINE_INCLUDE_DIR} 
                           ${OpenCV_INCLUDE_DIRS}
                           /usr/local/include)
target_link_libraries(${PROJECT_NAME} PRIVATE cudart NVCVImage VideoFX
                      bmf_module_sdk ${OpenCV_LIBS})
target_link_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../bmf/lib
                        ${CUDA_LIBRARY_DIR} ${MAXINE_LIBRARY_DIR} )
