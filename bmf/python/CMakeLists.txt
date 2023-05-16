file(GLOB PY_SRCS *.cpp *.h)

pybind11_add_module(_bmf SHARED ${PY_SRCS})

if(BMF_ENABLE_FFMPEG)
    target_link_libraries(_bmf PRIVATE nlohmann bmf_module_sdk engine ${BMF_FFMPEG_TARGETS})
else()
    target_link_libraries(_bmf PRIVATE nlohmann bmf_module_sdk engine)
endif()

if(BMF_ENABLE_CUDA)
    # need by ffmpeg for hwframe support
    target_link_libraries(_bmf
        PRIVATE cuda::cuda)
endif()


target_include_directories(_bmf PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
