# Define the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fPIC")

# Source files
set(LIB_SRCS_COPY copy_module.cpp)
set(MAIN_SRCS cpp_demo.cpp)

# Create shared libraries for copy and assemble modules
add_library(copy_module SHARED ${LIB_SRCS_COPY})

# Link the external libraries to the modules
target_link_libraries(copy_module engine bmf_module_sdk hmp)

# Create the executable and link the shared libraries
add_executable(cpp_demo ${MAIN_SRCS})
target_link_libraries(cpp_demo copy_module engine bmf_module_sdk hmp)

# Clean up build files
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "cpp_demo")
