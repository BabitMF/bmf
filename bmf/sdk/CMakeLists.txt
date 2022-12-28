
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_sdk/include/bmf/sdk/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_sdk/include/bmf/sdk/config.h")

file(GLOB CPP_SRCS cpp_sdk/src/*.cpp)
file(GLOB GO_SRCS go_sdk/src/*.cpp)
file(GLOB CPP_HDRS cpp_sdk/include/*.h)
file(GLOB GO_HDRS go_sdk/include/*.h)

if(BMF_ENABLE_MOBILE)
    list(FILTER CPP_SRCS EXCLUDE REGEX bmf_capi)
endif()

if(IOS)
add_library(bmf_module_sdk 
    STATIC 
    ${CPP_SRCS} ${CPP_HDRS}
)
else() 
add_library(bmf_module_sdk 
    ${CPP_SRCS} ${CPP_HDRS}
)
endif()


target_include_directories(bmf_module_sdk
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp_sdk/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/c_sdk/include>
        $<BUILD_INTERFACE:/usr/java/jdk1.8.0_301/include>
        $<BUILD_INTERFACE:/usr/java/jdk1.8.0_301/include/linux>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_sdk/src
)

target_link_libraries(bmf_module_sdk
    PUBLIC
        ${BMF_STD_DEPS}  
        hmp nlohmann
        #        -pie -fPIE -Wl,--export-dynamic
)

if(NOT APPLE AND NOT ANDROID)
    set_target_properties(bmf_module_sdk PROPERTIES
        VERSION "${BMF_VERSION_MAJOR}.${BMF_VERSION_MINOR}"
    )
endif()

target_compile_options(bmf_module_sdk PRIVATE
    -DBMF_BUILD_SHARED
    -fvisibility=hidden
)

if(BMF_ENABLE_MOBILE)
    target_compile_options(bmf_module_sdk PUBLIC
        -DNO_TRACE
    )
endif()


if(BMF_ENABLE_GLOG)
    target_link_libraries(bmf_module_sdk PUBLIC glog)
endif()


# tests
file(GLOB TEST_SRCS cpp_sdk/test/*.cpp)
add_executable(test_bmf_module_sdk ${TEST_SRCS})

target_link_libraries(test_bmf_module_sdk 
    PRIVATE
        ${BMF_FFMPEG_TARGETS} bmf_module_sdk
        gtest
)


if(BMF_ENABLE_CUDA)
    # need by ffmpeg for hwframe support
    target_link_libraries(test_bmf_module_sdk
        PRIVATE cuda::cuda)
endif()


target_link_libraries(test_bmf_module_sdk PRIVATE gtest_main)

if ($ENV{SCRIPT_EXEC_MODE} MATCHES "osx")
    # Mac platform
    set_target_properties(test_bmf_module_sdk
        PROPERTIES
        INSTALL_RPATH "@loader_path/../lib;@loader_path"
    )

    # Mac OSX excludes Python lib in linker to prevent embedding Python interpreter
    target_include_directories(test_bmf_module_sdk PRIVATE ${Python_INCLUDE_DIRS})
    set_target_properties(test_bmf_module_sdk PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

if(APPLE)
    set_target_properties(test_bmf_module_sdk
        PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "com.bytedance.test_bmf_module_sdk"
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.bytedance.test-bmf-module-sdk"
            MACOSX_BUNDLE_SHORT_VERSION_STRING ${BMF_BUILD_VERSION}
            MACOSX_BUNDLE_BUNDLE_VERSION ${BMF_BUILD_VERSION}
    )

endif()

install(TARGETS bmf_module_sdk test_bmf_module_sdk
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${CPP_HDRS} DESTINATION include/cpp/)