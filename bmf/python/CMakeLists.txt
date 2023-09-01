file(GLOB PY_SRCS *.cpp *.h)

pybind11_add_module(_bmf SHARED ${PY_SRCS})


if(BMF_ENABLE_FFMPEG)
    target_link_libraries(_bmf PRIVATE nlohmann bmf_module_sdk engine ${BMF_FFMPEG_TARGETS})
else()
    target_link_libraries(_bmf PRIVATE nlohmann bmf_module_sdk engine)
endif()

target_link_libraries(_bmf
        PRIVATE _hmp)

if(BMF_ENABLE_CUDA)
    # need by ffmpeg for hwframe support
    target_link_libraries(_bmf
        PRIVATE cuda::cuda)
endif()

if(BMF_ENABLE_TORCH)
    target_include_directories(_bmf PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(_bmf PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB})
endif()


target_include_directories(_bmf PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

if(WIN32)
    set_target_properties(_bmf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib
    )
endif()