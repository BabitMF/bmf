
# remove it to generate new jni header
# add_subdirectory(src/main/java)

add_library(_jhmp hmp_japi.cpp)
target_link_libraries(_jhmp PRIVATE 
                      hmp
                      ${JNI_LIBRARIES}
                     )

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(_jhmp PRIVATE pthread)
else()
    target_link_libraries(_jhmp PRIVATE jnigraphics)
endif()

target_include_directories(_jhmp PRIVATE
                          ${CMAKE_CURRENT_BINARY_DIR}/src/main/java
                          ${JNI_INCLUDE_DIRS}
                          ${CMAKE_CURRENT_SOURCE_DIR}
                          )

target_compile_options(_jhmp
    PRIVATE
        -fvisibility=hidden
)

# add_dependencies(_jhmp hmp_jni_compile)
