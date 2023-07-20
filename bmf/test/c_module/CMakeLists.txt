
file(GLOB SRCS *.cc *.h)

add_library(copy_module SHARED ${SRCS})

target_link_libraries(copy_module 
    PRIVATE 
    bmf_module_sdk
)

add_executable(test ${SRCS})

target_link_libraries(test
    PRIVATE 
        bmf_module_sdk 
        gtest
)

target_link_libraries(test PRIVATE gtest_main)
