cmake_minimum_required(VERSION 3.12)
project(cpp_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

SET(MODULE_SRCS ${PROJECT_SOURCE_DIR}/cpp_module.cpp)

add_library(cpp_module SHARED ${MODULE_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../../output/bmf/include
                           /usr/local/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../../output/bmf/lib)
target_link_libraries(cpp_module
    PRIVATE
    bmf_module_sdk
)

if(WIN32)
    set_target_properties(copy_module PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib
    )
endif()

#set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
#install(TARGETS cpp_module)
