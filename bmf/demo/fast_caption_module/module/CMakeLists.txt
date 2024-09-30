project(bmf_caption_modules)
set(CAPTION_LIB module)

add_library(${CAPTION_LIB} OBJECT)
file(GLOB SRCS *.cpp)
target_sources(${CAPTION_LIB} PUBLIC
${SRCS}
)

target_include_directories(${CAPTION_LIB} PUBLIC
    ${CMAKE_SOURCE_DIR}/tokenizer
    ${CMAKE_SOURCE_DIR}/libImaging
    ${CMAKE_SOURCE_DIR}/gpu
    ${BMF_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    /usr/local/cuda/include
    ${LIBTORCH_INCLUDE_DIR}
    ${LIBTORCH_INCLUDE_DIR}/torch/csrc/api/include
    ${TRT_LLM_INCLUDE_DIR}
)

add_definitions(-DHAVE_LIBJPEG)
set_property(TARGET ${CAPTION_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)