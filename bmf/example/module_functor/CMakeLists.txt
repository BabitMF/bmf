
add_executable(bmf_invoke bmf_invoke.cpp)

target_link_libraries(bmf_invoke PRIVATE bmf_module_sdk)

target_include_directories(bmf_invoke 
		PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})