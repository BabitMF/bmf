
file(GLOB SRCS src/*.cpp
               ../connector/src/*.cpp
               ../connector/internal/*.cpp
               ../connector/internal/*.hpp
)
file(GLOB HDRS include/*.h ../connector/include/*.h)

if(BMF_ENABLE_MOBILE)
    list(FILTER SRCS EXCLUDE REGEX connector_capi)
endif()

add_library(engine ${SRCS} ${HDRS})


target_include_directories(engine
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bmf/engine/c_engine/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bmf/engine/connector/include>
    PRIVATE
        src
)
if(BMF_ENABLE_BREAKPAD) 
    add_definitions(-DBMF_ENABLE_BREAKPAD)
    target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/3rd_party/breakpad/include/breakpad)
    target_link_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/3rd_party/breakpad/lib)
    target_link_libraries(engine PUBLIC libbreakpad_client.a)
endif()
target_link_libraries(engine
    PRIVATE 
        bmf_module_sdk
)

if(NOT APPLE AND NOT ANDROID)
    set_target_properties(engine PROPERTIES
        VERSION "${BMF_VERSION_MAJOR}.${BMF_VERSION_MINOR}"
    )
endif()

add_custom_command(TARGET engine
    POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/BUILTIN_CONFIG.json $<TARGET_FILE_DIR:engine>/../)


# tests
file(GLOB TEST_SRCS test/*.cpp)

list(FILTER TEST_SRCS EXCLUDE REGEX ".*test_go_module.cpp")

add_executable(test_bmf_engine ${TEST_SRCS})

target_link_libraries(test_bmf_engine
    PRIVATE 
        ${BMF_FFMPEG_TARGETS}
        bmf_module_sdk engine
        gtest
)

target_link_libraries(test_bmf_engine PRIVATE gtest_main)

if ($ENV{SCRIPT_EXEC_MODE} MATCHES "osx")
    # Mac platform
    set_target_properties(test_bmf_engine
        PROPERTIES
        INSTALL_RPATH "@loader_path/../lib;@loader_path"
    )

    # Mac OSX excludes Python lib in linker to prevent embedding Python interpreter
    target_include_directories(test_bmf_engine PRIVATE ${Python_INCLUDE_DIRS})
    set_target_properties(test_bmf_engine PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

add_custom_command(TARGET test_bmf_engine
    POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_directory 
            ${CMAKE_CURRENT_SOURCE_DIR}/files $<TARGET_FILE_DIR:test_bmf_engine>/../files)

add_custom_command(TARGET test_bmf_engine
    POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy 
            ${CMAKE_CURRENT_SOURCE_DIR}/BUILTIN_CONFIG.json $<TARGET_FILE_DIR:test_bmf_engine>/../)

add_custom_command(TARGET test_bmf_engine
    POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy 
            ${CMAKE_CURRENT_SOURCE_DIR}/BUILTIN_CONFIG.json ${BMF_ASSEMBLE_ROOT}/bmf)


if(BMF_ENABLE_PYTHON)
    add_library(bmf_py_loader SHARED src/loader/py_module_loader.cpp)
    target_link_libraries(bmf_py_loader 
        PRIVATE 
        pybind11::pybind11 bmf_module_sdk Python::Python)
endif()

add_library(bmf_go_loader SHARED src/loader/go_module_loader.cpp)
target_link_libraries(bmf_go_loader 
    PRIVATE 
    bmf_module_sdk)



install(TARGETS engine 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)