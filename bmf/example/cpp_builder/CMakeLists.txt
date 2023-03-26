
file(GLOB SRCS *.cpp *.h)

add_executable(test_cpp_builder ${SRCS})

target_link_libraries(test_cpp_builder
    PRIVATE 
        bmf_module_sdk engine nlohmann 
        gtest
)

target_link_libraries(test_cpp_builder PRIVATE gtest_main)

install(TARGETS test_cpp_builder
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)
