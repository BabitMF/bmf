file(GLOB SRCS *.cc *.h)

add_library(copy_module SHARED ${SRCS})
set_property(TARGET PROPERTY CXX_STANDARD 17)

target_link_libraries(copy_module
    PRIVATE
    bmf_module_sdk
)

#set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
#install(TARGETS copy_module)
