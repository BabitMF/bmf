file(GLOB PY_SRCS *.cpp *.h)

pybind11_add_module(_bmf SHARED ${PY_SRCS})

target_link_libraries(_bmf PRIVATE nlohmann bmf_module_sdk engine)

target_include_directories(_bmf PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
