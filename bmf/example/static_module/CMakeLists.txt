
add_library(test_module test_module.cpp)

target_include_directories(test_module
    PRIVATE
    ${IOS_ASSEMBLE_ROOT}/include
)

add_executable(test_module_main test_module_main.cpp)

target_include_directories(test_module_main PRIVATE
	${IOS_ASSEMBLE_ROOT}/include)

target_link_libraries(test_module_main
    PRIVATE hmp fmt bmf_module_sdk test_module
)

if(APPLE)
    set_target_properties(test_module_main
        PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "com.bytedance.test_module_main"
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.bytedance.test-module-main"
            MACOSX_BUNDLE_SHORT_VERSION_STRING 1.0.0 
            MACOSX_BUNDLE_BUNDLE_VERSION 1.0.0
    )
endif()

add_dependencies(test_module_main bmf_assem test_module)
add_dependencies(test_module bmf_assem test_module)