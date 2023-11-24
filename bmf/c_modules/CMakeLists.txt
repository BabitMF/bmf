function(mac_update)
    if ($ENV{SCRIPT_EXEC_MODE} MATCHES "osx")
        # Mac platform
        set_target_properties(${ARGV0}
            PROPERTIES
            INSTALL_RPATH "@loader_path/../lib;@loader_path"
        )

        # Mac OSX excludes Python lib in linker to prevent embedding Python interpreter
        target_include_directories(${ARGV0} PRIVATE ${Python_INCLUDE_DIRS})
        set_target_properties(${ARGV0} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
endfunction()

function(set_soname)
    if(NOT APPLE AND NOT ANDROID)
        set_target_properties(${ARGV0} PROPERTIES
            VERSION "${BMF_VERSION_MAJOR}.${BMF_VERSION_MINOR}.${BMF_VERSION_PATCH}"
            SOVERSION "${BMF_VERSION_MAJOR}"
        )
    endif()
endfunction()

function(module_install)
    if(WIN32)
        set_target_properties(${ARGV0} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/cpp_modules/Module_${ARGV0}
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/cpp_modules/Module_${ARGV0}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/cpp_modules/Module_${ARGV0}
        )
    else()
        set_target_properties(${ARGV0} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/cpp_modules/Module_${ARGV0})
    endif()

    # copy meta.info
    add_custom_command(TARGET ${ARGV0}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/meta/${ARGV0}.info ${BMF_ASSEMBLE_ROOT}/bmf/cpp_modules/Module_${ARGV0}/meta.info
    )
endfunction()

if(BMF_ENABLE_FFMPEG)
    set(HDRS include/ffmpeg_decoder.h
             include/ffmpeg_encoder.h
             include/ffmpeg_filter.h
             include/c_module.h
             include/video_sync.h
             include/av_common_utils.h
             include/audio_fifo.h
             include/audio_resampler.h
    )
    set(SRCS src/ffmpeg_decoder.cpp
             src/ffmpeg_encoder.cpp
             src/ffmpeg_filter.cpp
             src/ffmpeg_func_registry.cpp
             src/video_sync.cpp
             src/audio_fifo.cpp
             src/audio_resampler.cpp
    )

    add_library(builtin_modules SHARED ${SRCS} ${HDRS})
    target_include_directories(builtin_modules
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bmf/c_modules/include>
    )
    if(WIN32)
        set_target_properties(builtin_modules PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib
        )
    endif()
    if(BMF_ENABLE_CUDA)
        # need by ffmpeg for hwframe support
        target_link_libraries(builtin_modules
            PRIVATE cuda::cuda)
    endif()

    target_link_libraries(builtin_modules
        PRIVATE
            ${BMF_FFMPEG_TARGETS}
            bmf_module_sdk
    )

    set_soname(builtin_modules)
    mac_update(builtin_modules)


    add_custom_command(TARGET builtin_modules
        POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/meta/BUILTIN_CONFIG.json $<TARGET_FILE_DIR:builtin_modules>/../)

    install(TARGETS builtin_modules
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )

    install(FILES ${HDRS} DESTINATION include)
endif()

# pass_through module
set(PASS_THROUGH_HDRS include/pass_through_module.h)
set(PASS_THROUGH_SRCS src/pass_through_module.cpp)
add_library(pass_through SHARED ${PASS_THROUGH_HDRS} ${PASS_THROUGH_SRCS})
target_include_directories(pass_through PUBLIC include)
target_link_libraries(pass_through PRIVATE bmf_module_sdk)
set_soname(pass_through)
mac_update(pass_through)
module_install(pass_through)

# clock module
set(CLOCK_MODULE_HDRS include/clock_module.h include/fraction.hpp)
set(CLOCK_MODULE_SRCS src/clock_module.cpp)
add_library(clock SHARED ${CLOCK_MODULE_HDRS} ${CLOCK_MODULE_SRCS})
target_include_directories(clock PUBLIC include)
target_link_libraries(clock PRIVATE bmf_module_sdk)
set_soname(clock)
mac_update(clock)
module_install(clock)

# MockDecoder
set(MOCK_DECODER_MODULE_HDRS include/mock_decoder.h)
set(MOCK_DECODER_MODULE_SRCS src/mock_decoder.cpp)
add_library(MockDecoder SHARED ${MOCK_DECODER_MODULE_HDRS} ${MOCK_DECODER_MODULE_SRCS})
target_include_directories(MockDecoder PUBLIC include)
target_link_libraries(MockDecoder PRIVATE bmf_module_sdk)
set_soname(MockDecoder)
mac_update(MockDecoder)
module_install(MockDecoder)

# GoMockDecoder
set(GOMOCK_DECODER_MODULE_HDRS include/go_mock_decoder.h)
set(GOMOCK_DECODER_MODULE_SRCS src/go_mock_decoder.cpp)
add_library(GoMockDecoder SHARED ${GOMOCK_DECODER_MODULE_HDRS} ${GOMOCK_DECODER_MODULE_SRCS})
target_include_directories(GoMockDecoder PUBLIC include)
target_link_libraries(GoMockDecoder PRIVATE bmf_module_sdk)
set_soname(GoMockDecoder)
mac_update(GoMockDecoder)
module_install(GoMockDecoder)

# tests, commented
if(BMF_ENABLE_TEST)
    #file(GLOB TEST_SRCS test/*.cpp)

    ## compile errors
    #list(FILTER TEST_SRCS EXCLUDE REGEX test_python_module.cpp)

    #add_executable(test_builtin_modules ${TEST_SRCS})

    #target_link_libraries(test_builtin_modules
    #    PRIVATE
    #        builtin_modules engine bmf_module_sdk
    #        gtest ${BMF_FFMPEG_TARGETS}
    #)

    #target_link_libraries(test_builtin_modules PRIVATE gtest_main)

    #mac_update(test_builtin_modules)
endif()
