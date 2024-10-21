

file(GLOB PY_SRCS *.cpp *.h)

if (APPLE AND NOT HMP_LOCAL_DEPENDENCIES)
    pybind11_add_module(_hmp MODULE ${PY_SRCS})
else()
    pybind11_add_module(_hmp SHARED ${PY_SRCS})
endif()

target_link_libraries(_hmp PRIVATE hmp)

if(WIN32)
    set_target_properties(_hmp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib
    )
endif()


if(HMP_ENABLE_TORCH)
    target_link_libraries(_hmp PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB})
endif()

if(HMP_ENABLE_CUDA) # need by ffmpeg.h
    target_include_directories(_hmp PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

if(BUILD_SHARED_LIBS)
    target_compile_options(_hmp
        PRIVATE
            -DHMP_BUILD_SHARED
        )
endif()


target_include_directories(_hmp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    )


# tests
file(GLOB PY_TEST_SRCS ${PROJECT_SOURCE_DIR}/tests/test_*)

foreach(PY_TEST ${PY_TEST_SRCS})
    get_filename_component(TEST_NAME ${PY_TEST} NAME_WE)
    add_test(NAME ${TEST_NAME}
        COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY} 
            pytest -v ${PY_TEST}
        )
endforeach()
