
file(GLOB SRCS *.cpp *.h)

find_package(OpenSSL 3.0 QUIET)
if(NOT OpenSSL_FOUND)
    find_package(OpenSSL REQUIRED)
endif()

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
        OpenSSL::Crypto OpenSSL::SSL
    )
else()
    target_link_libraries(test_cpp_builder
        PRIVATE
        OpenSSL::Crypto OpenSSL::SSL
    )
endif()

target_link_libraries(test_cpp_builder PRIVATE gtest_main)

install(TARGETS test_cpp_builder
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)
