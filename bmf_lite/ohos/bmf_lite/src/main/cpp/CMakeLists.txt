# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5)
project(BmfLite)

set(BMF_LITE_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${BMF_LITE_ROOT_PATH}
                    ${BMF_LITE_ROOT_PATH}/include)

add_library(bmf_lite_static SHARED bmf_lite_init.cpp)

add_subdirectory(../../../../../src ../../../bmf_lite/)

target_link_libraries(bmf_lite_static PUBLIC
   libace_napi.z.so
   EGL
   GLESv3
   hilog_ndk.z
   ace_ndk.z
   ace_napi.z
   libc++.a
)
