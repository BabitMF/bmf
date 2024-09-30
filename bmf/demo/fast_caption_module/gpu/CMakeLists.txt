project(bmf_caption_modules)
set(GPU_LIB gpu)

add_library(${GPU_LIB} OBJECT)
file(GLOB SRCS *.cpp)
target_sources(${GPU_LIB} PUBLIC ${SRCS})
target_include_directories(${GPU_LIB} PUBLIC 
    /usr/local/cuda/include
)
set_property(TARGET ${GPU_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)