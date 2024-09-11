cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

cmake_policy(SET CMP0077 NEW)

project(HML CXX C)

### general settings
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

### options

option(BMF_LITE_ENABLE_CVPIXELBUFFER "Enable build CVpixelBuffer" OFF)
option(BMF_LITE_ENABLE_METALBUFFER "Enable build Metal texture buffer" OFF)
option(BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER "Enable build opengl texture buffer" OFF)
option(BMF_LITE_ENABLE_CPUMEMORYBUFFER "Enable build cpu memory buffer" OFF)
option(BMF_LITE_ENABLE_BUFFER_TRANSFORM "Enable build buffer transform" OFF)

option(BMF_LITE_ENABLE_SUPER_RESOLUTION "Enable build super resolution" OFF)
option(BMF_LITE_ENABLE_DENOISE "Enable build denoise" OFF)
option(BMF_LITE_ENABLE_CANNY "Enable build canny" OFF)
option(BMF_LITE_ENABLE_TEX_GEN_PIC "Enable build tex_gen_pic" OFF)

option(BMF_ENABLE_LOG "Enable log" ON)

add_subdirectory(src)

add_subdirectory(demo)