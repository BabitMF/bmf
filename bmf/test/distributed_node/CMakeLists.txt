# Define the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fPIC")

# Include directories
include_directories(
    /root/workspace/bmf_OSPP/output/bmf/include
    .
)

# Link directories
link_directories(/root/workspace/bmf_OSPP/output/bmf/lib)

# Source files
set(LIB_SRCS_SPLIT split_module.cpp)
set(LIB_SRCS_COPY copy_module.cpp)
set(LIB_SRCS_ASSEMBLE assemble_module.cpp)
set(MAIN_SRCS cpp_demo.cpp)

# Create shared libraries for copy and assemble modules
add_library(split_module SHARED ${LIB_SRCS_SPLIT})
add_library(copy_module SHARED ${LIB_SRCS_COPY})
add_library(assemble_module SHARED ${LIB_SRCS_ASSEMBLE})

# Link the external libraries to the modules
target_link_libraries(split_module engine bmf_module_sdk)
target_link_libraries(copy_module engine bmf_module_sdk)
target_link_libraries(assemble_module engine bmf_module_sdk)

# Create the executable and link the shared libraries
add_executable(cpp_demo ${MAIN_SRCS})
target_link_libraries(cpp_demo split_module copy_module assemble_module engine bmf_module_sdk)

# Clean up build files
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "cpp_demo")
