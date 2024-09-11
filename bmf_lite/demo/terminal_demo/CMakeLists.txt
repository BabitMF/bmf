cmake_policy(SET CMP0074 NEW)

set(MiniOpenCV_BUILD_VERSION 1.0.0.15)
if(ANDROID)
    message("terminal_demo android")
    add_executable(test_bmf_lite_android_interface test_android_interface.cpp)
    target_include_directories(test_bmf_lite_android_interface PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../api/cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../api/cpp/algorithm
                    )
    target_link_libraries(test_bmf_lite_android_interface PRIVATE
        ${MiniOpenCV_LIBS}
        EGL
        android
        jnigraphics
        bmf_lite
        GLESv3
        log
    )
endif()