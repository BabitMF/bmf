file(GLOB SRCS *.cc *.h)

add_library(copy_module SHARED ${SRCS})
set_property(TARGET PROPERTY CXX_STANDARD 17)

target_link_libraries(copy_module
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
#install(TARGETS copy_module)
