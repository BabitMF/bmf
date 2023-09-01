
file(GLOB SRCS *.cpp *.h)

add_executable(test_cpp_builder ${SRCS})

target_link_libraries(test_cpp_builder
    PRIVATE 
        bmf_module_sdk engine nlohmann 
        gtest
)

if(WIN32)
    target_link_directories(test_cpp_builder PRIVATE $ENV{WIN_XCOMPILE_ROOT}/$ENV{WIN_XCOMPILE_ARCH}/usr/lib)
    target_link_libraries(test_cpp_builder
        PRIVATE
        libcrypto libssl
    )
else()
    target_link_libraries(test_cpp_builder
        PRIVATE
        crypto ssl
    )
endif()

target_link_libraries(test_cpp_builder PRIVATE gtest_main)

install(TARGETS test_cpp_builder
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)
