cmake_minimum_required(VERSION 3.12)
project(hw_frame_gen)

file(GLOB SRCS *.cc *.h)

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

set(BMF_ROOT "/mnt/bmf-v2-compile/bmf/output" CACHE STRING "Bmf output path")
set(FFMPEG_ROOT "/mnt/bmf-v2-compile/bmf/3rd_party/ffmpeg_bin/linux/build/" CACHE STRING "ffmpeg output path")
set(BMF_INCLUDE_PATH ${BMF_ROOT}/bmf/include)
set(BMF_LIB_PATH ${BMF_ROOT}/bmf/lib/)
find_library(BMF_LIB bmf_module_sdk PATHS ${BMF_LIB_PATH})
message("Find BMF SDK Library: ${BMF_LIB}")

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_path(AVUTIL_INCLUDE_DIR libavutil/frame.h)
find_library(AVCODEC_LIBRARY avcodec)
find_library(AVUTIL_LIBRARY avutil)
message("Find FFMPEG include DIR: ${AVCODEC_INCLUDE_DIR}, lib: ${AVCODEC_LIBRARY}, ${AVUTIL_LIBRARY}")

add_library(hw_frame_gen SHARED ${SRCS})
target_include_directories(hw_frame_gen
        PRIVATE
        ${AVCODEC_INCLUDE_DIR}
        ${AVUTIL_INCLUDE_DIR}
        ${BMF_INCLUDE_PATH}
        /usr/local/cuda/targets/x86_64-linux/include
        )
target_link_libraries(hw_frame_gen
    PRIVATE 
    ${BMF_LIB}
    ${AVCODEC_LIBRARY}
    ${AVUTIL_LIBRARY}
)

#target_link_libraries(test PRIVATE gtest_main)
