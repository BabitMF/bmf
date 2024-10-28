# Recipe to build copy module
add_library(copy_module SHARED 
    copy_module.cc
    copy_module.h
)
target_link_libraries(copy_module
    PRIVATE
    bmf_module_sdk
)
set_property(TARGET copy_module PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
target_link_options(copy_module PRIVATE "-sSIDE_MODULE=2")
target_compile_options(copy_module PRIVATE "-sSIDE_MODULE=2")


# Gernal flag
set(DEMO_FLAGS "-fexceptions -s ASYNCIFY -s 'ASYNCIFY_IMPORTS=[\"loadLibrary\"]'")
list(APPEND DEMO_FLAGS "-s ASYNCIFY_STACK_SIZE=5120000")
list(APPEND DEMO_FLAGS "-s MAIN_MODULE=1")
list(APPEND DEMO_FLAGS "-s USE_PTHREADS=1")
list(APPEND DEMO_FLAGS "-s SHARED_MEMORY=1")
list(APPEND DEMO_FLAGS "-s INITIAL_MEMORY=64MB")
list(APPEND DEMO_FLAGS "-s ALLOW_MEMORY_GROWTH")
list(APPEND DEMO_FLAGS "--preload-file ${CMAKE_SOURCE_DIR}/bmf/c_modules/meta/BUILTIN_CONFIG.json@BUILTIN_CONFIG.json")
list(APPEND DEMO_FLAGS "--preload-file ${CMAKE_SOURCE_DIR}/big_bunny_10s_30fps.mp4@big_bunny_10s_30fps.mp4")
string(REPLACE ";" " " DEMO_FLAGS "${DEMO_FLAGS}")

# Demo1: builtin modules with video
add_executable(demo_builtin_video demo_builtin_video.cc)
set(CMAKE_EXECUTABLE_SUFFIX ".html")
set_target_properties(demo_builtin_video PROPERTIES LINK_FLAGS "${DEMO_FLAGS}")
target_link_libraries(demo_builtin_video
    PRIVATE 
    engine bmf_module_sdk
)

# Demo2: copy module
add_executable(demo_copy_module demo_copy_module.cc)
set(CMAKE_EXECUTABLE_SUFFIX ".html")
set_target_properties(demo_copy_module PROPERTIES LINK_FLAGS "${DEMO_FLAGS}")
target_link_libraries(demo_copy_module
    PRIVATE 
    engine bmf_module_sdk
)