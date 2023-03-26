
file(GLOB HDRS include/*.h)
if(BMF_ENABLE_FFMPEG)
    file(GLOB SRCS src/*.cpp)
else()
    set(SRCS src/pass_through_module.cpp
             src/clock_module.cpp
        )
endif()

add_library(builtin_modules SHARED ${SRCS} ${HDRS})

target_include_directories(builtin_modules
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bmf/c_modules/include>
)

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

if(NOT APPLE AND NOT ANDROID)
    set_target_properties(builtin_modules PROPERTIES
        VERSION "${BMF_VERSION_MAJOR}.${BMF_VERSION_MINOR}"
    )
endif()

if ($ENV{SCRIPT_EXEC_MODE} MATCHES "osx")
    # Mac platform
    set_target_properties(builtin_modules
        PROPERTIES
        INSTALL_RPATH "@loader_path/../lib;@loader_path"
    )

    # Mac OSX excludes Python lib in linker to prevent embedding Python interpreter
    target_include_directories(builtin_modules PRIVATE ${Python_INCLUDE_DIRS})
    set_target_properties(builtin_modules PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

# tests
file(GLOB TEST_SRCS test/*.cpp)

# compile errors
list(FILTER TEST_SRCS EXCLUDE REGEX test_python_module.cpp)

add_executable(test_builtin_modules ${TEST_SRCS})

target_link_libraries(test_builtin_modules 
    PRIVATE
        builtin_modules engine bmf_module_sdk
        gtest ${BMF_FFMPEG_TARGETS}
)

target_link_libraries(test_builtin_modules PRIVATE gtest_main)

if ($ENV{SCRIPT_EXEC_MODE} MATCHES "osx")
    # Mac platform
    set_target_properties(test_builtin_modules
        PROPERTIES
        INSTALL_RPATH "@loader_path/../lib;@loader_path"
    )

    # Mac OSX excludes Python lib in linker to prevent embedding Python interpreter
    target_include_directories(test_builtin_modules PRIVATE ${Python_INCLUDE_DIRS})
    set_target_properties(test_builtin_modules PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

install(TARGETS builtin_modules
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(FILES ${HDRS} DESTINATION include)
