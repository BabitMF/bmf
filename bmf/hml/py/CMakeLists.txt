

file(GLOB PY_SRCS *.cpp *.h)

pybind11_add_module(_hmp MODULE ${PY_SRCS})

target_link_libraries(_hmp PRIVATE hmp)

if(HMP_ENABLE_TORCH)
    target_link_libraries(_hmp PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB})
endif()

if(HMP_ENABLE_CUDA) # need by ffmpeg.h
    target_include_directories(_hmp PRIVATE ${CUDA_INCLUDE_DIRS})
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
