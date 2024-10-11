cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-exceptions -fno-rtti")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fdata-sections -ffunction-sections")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fdata-sections -ffunction-sections")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Oz -flto -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Oz -flto -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -O2 -flto -Wl,--exclude-libs,ALL -Wl,--gc-sections")

message( "Android Platform")
option(BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER "Enable Enable build OpenglBuffer" ON)
option(BMF_LITE_ENABLE_CPUMEMORYBUFFER "Enable build CpuMemoryBuffer" ON)
option(BMF_LITE_ENABLE_DENOISE "Enable build denoise" ON)
option(BMF_LITE_ENABLE_SUPER_RESOLUTION "Enable build super resolution" ON)
option(BMF_LITE_ENABLE_TEX_GEN_PIC "Enable build tex_gen_pic" ON)
option(BMF_ENABLE_JNI "Enable build with JNI support" ON)

if(BMF_ENABLE_JNI)
    add_compile_definitions(BMF_LITE_ENABLE_ALGORITHM)
    add_compile_definitions(BMF_LITE_ENABLE_PARAM)
    add_compile_definitions(BMF_LITE_ENABLE_VIDEO_FRAME)
endif()

file(GLOB_RECURSE JNI_SRC main/cpp/*.cpp)
list(APPEND SRC ${JNI_SRC})
add_library(bmf_lite_jni SHARED ${SRC})

target_link_libraries(bmf_lite_jni PRIVATE EGL log android)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(bmf_lite_jni PRIVATE pthread)
endif()

message("Android CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/main/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../api/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../api/cpp/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../src/utils)
target_link_libraries(bmf_lite_jni
        PUBLIC
        bmf_lite
        EGL GLESv3 log
        )

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../bmf_lite/src bmf_lite)