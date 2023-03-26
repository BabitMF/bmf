cmake_minimum_required(VERSION 3.13)
project(custom_ffmpeg)
set(CMAKE_CXX_STANDARD 17)

## options
set(BMF_ROOT "/mnt/bmf-v2-compile/bmf/output" CACHE STRING "Bmf output path")
set(FFMPEG_ROOT "/mnt/bmf-v2-compile/bmf/3rd_party/ffmpeg_bin/linux/build/" CACHE STRING "ffmpeg output path")

## BMF setup
set(BMF_INCLUDE_PATH ${BMF_ROOT}/bmf/include)
set(BMF_LIB_PATH ${BMF_ROOT}/bmf/lib/)
find_library(BMF_LIB bmf_module_sdk PATHS ${BMF_LIB_PATH})
message("Find BMF SDK Library: ${BMF_LIB}")

## FFMPEG setup
#find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h REQUIRED
#        PATHS ${FFMPEG_ROOT} /opt/conda/include /usr/include /usr/local/include ${FFMPEG_ROOT}/include)

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)
find_path(AVFILTER_INCLUDE_DIR libavfilter/avfilter.h)
find_library(AVFILTER_LIBRARY avfilter)

message("Find FFMPEG include DIR: ${AVCODEC_INCLUDE_DIR}")

##
add_library(custom_ffmpeg SHARED custom_ffmpeg.cpp)

include_directories("./include")
target_include_directories(custom_ffmpeg
        PRIVATE
        ${BMF_INCLUDE_PATH}
        ${AVCODEC_INCLUDE_DIR}
        ${AVFILTER_INCLUDE_DIR}
        /usr/local/cuda/targets/x86_64-linux/include
        )
target_link_libraries(custom_ffmpeg PRIVATE ${BMF_LIB} ${AVFILTER_LIBRARY})
target_compile_definitions(custom_ffmpeg PRIVATE -DFMT_HEADER_ONLY)
