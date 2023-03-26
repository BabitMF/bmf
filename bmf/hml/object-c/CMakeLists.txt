
file(GLOB OC_SRCS src/*.mm)
file(GLOB OC_HDRS include/hmp/oc/*.h)

add_library(hmp_oc ${OC_SRCS} ${OC_HDRS})

target_include_directories(hmp_oc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

target_link_libraries(hmp_oc
    PRIVATE hmp
    PUBLIC
        "-framework CoreVideo"
    )

file(GLOB TEST_SRCS tests/*.mm)
add_executable(test_hmp_oc ${TEST_SRCS})

target_link_libraries(test_hmp_oc 
    PRIVATE 
        hmp_oc
        hmp
        "-framework Metal"
        "-framework OpenGLES"
    )
target_include_directories(test_hmp_oc PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/tests
     ../include)


if(APPLE)
    set_target_properties(test_hmp_oc
        PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "com.bytedance.test_hmp_oc"
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.bytedance.test-hmp-sdk-oc"
            MACOSX_BUNDLE_SHORT_VERSION_STRING ${HMP_BUILD_VERSION}
            MACOSX_BUNDLE_BUNDLE_VERSION ${HMP_BUILD_VERSION}
    )
endif()