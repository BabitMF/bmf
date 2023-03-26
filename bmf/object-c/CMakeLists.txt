
#ios wrapper
file(GLOB IOS_OC_SRCS src/*.mm)
file(GLOB IOS_OC_HEADERS include/bmf/oc/*.h)

add_library(bmf_module_sdk_oc ${IOS_OC_SRCS} ${IOS_OC_HEADERS})

target_include_directories(bmf_module_sdk_oc 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/bmf/hml/object-c/include>
    )

target_link_libraries(bmf_module_sdk_oc
    PRIVATE bmf_module_sdk hmp_oc)

file(GLOB TEST_SRCS tests/*.mm)
add_executable(test_bmf_module_sdk_oc ${TEST_SRCS})

target_link_libraries(test_bmf_module_sdk_oc PRIVATE bmf_module_sdk_oc)

target_include_directories(test_bmf_module_sdk_oc PRIVATE
        ${PROJECT_SOURCE_DIR}/bmf/hml/object-c/include
)

if(APPLE)
    set_target_properties(test_bmf_module_sdk_oc
        PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "com.bytedance.test_bmf_module_sdk_oc"
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.bytedance.test-bmf-module-sdk-oc"
            MACOSX_BUNDLE_SHORT_VERSION_STRING ${BMF_BUILD_VERSION}
            MACOSX_BUNDLE_BUNDLE_VERSION ${BMF_BUILD_VERSION}
    )
endif()