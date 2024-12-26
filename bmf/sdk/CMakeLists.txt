
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_sdk/include/bmf/sdk/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_sdk/include/bmf/sdk/config.h")

file(GLOB CPP_SRCS cpp_sdk/src/*.cpp)
file(GLOB GO_SRCS go_sdk/src/*.cpp)
file(GLOB CPP_HDRS cpp_sdk/include/*.h)
file(GLOB GO_HDRS go_sdk/include/*.h)

if($ENV{SCRIPT_EXEC_MODE} MATCHES "win")
    file(GLOB DLSRCS
        $ENV{WIN_XCOMPILE_ROOT}/$ENV{WIN_XCOMPILE_ARCH}/usr/src/dlfcn.c
        $ENV{WIN_XCOMPILE_ROOT}/$ENV{WIN_XCOMPILE_ARCH}/usr/include/dlfcn.h
    )
endif()

if(BMF_ENABLE_MOBILE)
    list(FILTER CPP_SRCS EXCLUDE REGEX bmf_capi)
endif()

if(IOS OR EMSCRIPTEN)
    add_library(bmf_module_sdk
        STATIC
        ${CPP_SRCS} ${CPP_HDRS}
    )
    if (EMSCRIPTEN)
        # Currently,we need async feature to open external library larger than 8MB.
        set_target_properties(bmf_module_sdk PROPERTIES LINK_FLAGS "-s ASYNCIFY -s 'ASYNCIFY_IMPORTS=[\"loadLibrary\"]'")
    endif()
else()
    add_library(bmf_module_sdk
        ${CPP_SRCS} ${CPP_HDRS}
    )
endif()

if(WIN32)
    set_target_properties(bmf_module_sdk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib
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

if($ENV{SCRIPT_EXEC_MODE} MATCHES "win")
    target_include_directories(bmf_module_sdk
        PUBLIC
            $ENV{WIN_XCOMPILE_ROOT}/$ENV{WIN_XCOMPILE_ARCH}/usr/include
    )
    target_link_directories(bmf_module_sdk PRIVATE $ENV{WIN_XCOMPILE_ROOT}/$ENV{WIN_XCOMPILE_ARCH}/usr/lib)
endif()


target_link_libraries(bmf_module_sdk
    PUBLIC
    ${BMF_STD_DEPS}
     hmp nlohmann
)

if(NOT APPLE AND NOT ANDROID)
    set_target_properties(bmf_module_sdk PROPERTIES
	VERSION "${BMF_VERSION_MAJOR}.${BMF_VERSION_MINOR}.${BMF_VERSION_PATCH}"
	SOVERSION "${BMF_VERSION_MAJOR}"
    )
endif()

target_compile_definitions(bmf_module_sdk PRIVATE BMF_BUILD_SHARED_SDK)

set_target_properties(bmf_module_sdk PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
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
if(BMF_ENABLE_TEST)
    file(GLOB TEST_SRCS cpp_sdk/test/*.cpp)

    if(FUZZTEST_ENABLE_FUZZING_MODE)
        fuzztest_setup_fuzzing_flags()
    endif()

    add_executable(test_bmf_module_sdk ${TEST_SRCS})

    target_link_libraries(test_bmf_module_sdk
        PRIVATE
            ${BMF_FFMPEG_TARGETS} bmf_module_sdk
            gtest
    )

    if(BMF_ENABLE_FUZZTEST)
        link_fuzztest(test_bmf_module_sdk)
    endif()

    if(HMP_ENABLE_OPENCV)
        # need by ffmpeg for hwframe support
        target_link_libraries(test_bmf_module_sdk
            PUBLIC hmp)
    endif()


    if(BMF_ENABLE_CUDA)
        # need by ffmpeg for hwframe support
        target_link_libraries(test_bmf_module_sdk
            PRIVATE cuda::cuda)
    endif()

    if(BMF_ENABLE_TORCH)
            target_include_directories(test_bmf_module_sdk PUBLIC ${TORCH_INCLUDE_DIRS})
            target_link_libraries(test_bmf_module_sdk
                PRIVATE ${TORCH_LIBRARIES})
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

    install(TARGETS test_bmf_module_sdk
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(TARGETS bmf_module_sdk
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${CPP_HDRS} DESTINATION include/cpp/)
