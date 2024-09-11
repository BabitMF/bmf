# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5)
project(BmfLite)

set(NATIVE_RENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DOHOS_PLATFORM)

set(BMF_LITE_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

message(STATUS "OHOS add_subdirectory")

add_subdirectory(${BMF_LITE_ROOT_PATH}/src ../../../bmf_lite)

option(BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER "Enable build opengl texture buffer" ON)
option(BMF_LITE_ENABLE_SUPER_RESOLUTION "Enable build super resolution" ON)
option(BMF_LITE_ENABLE_DENOISE "Enable build super resolution" ON)
option(BMF_ENABLE_LOG "Enable log" ON)

list(APPEND NATIVE_RENDER_SRC
    native_render_init.cpp
    manager/plugin_manager.cpp
    render/egl_render_context.cpp
    render/plugin_render.cpp
    render/render_thread.cpp
    render/shader_program.cpp
    render/oes_to_2d_renderer.cpp
    render/split_screen_renderer.cpp
    camera/camera_manager.cpp
    player/player_manager.cpp
    algorithm/algorithm.cpp
)

add_library(nativerender SHARED
    ${NATIVE_RENDER_SRC}
)

target_link_libraries(nativerender PRIVATE
    bmf_lite
)

target_include_directories(nativerender PUBLIC
    ${NATIVE_RENDER_ROOT_PATH}
    ${NATIVE_RENDER_ROOT_PATH}/include
    ${NATIVE_RENDER_ROOT_PATH}/algorithm
    ${NATIVE_RENDER_ROOT_PATH}/common
    ${BMF_LITE_ROOT_PATH}/api/cpp
    ${BMF_LITE_ROOT_PATH}/src
)

target_link_libraries(nativerender PUBLIC
   EGL
   GLESv3
   hilog_ndk.z
   ace_ndk.z
   ace_napi.z
   native_vsync
   native_drawing
   native_window
   ohcamera
   native_image
   avplayer
)
