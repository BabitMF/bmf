project(bmf_caption_modules)
set(TOKENIZER_LIB tokenizer)

add_library(${TOKENIZER_LIB} OBJECT)
target_sources(${TOKENIZER_LIB} PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/tokenizer.cpp
)
target_include_directories(${TOKENIZER_LIB} PUBLIC
${BMF_INCLUDE_DIR}
)
set_property(TARGET ${TOKENIZER_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)