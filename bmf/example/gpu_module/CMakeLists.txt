
file(GLOB SRCS *.cpp *.h)

add_library(cvtcolor SHARED ${SRCS})

target_link_libraries(cvtcolor 
    PRIVATE 
    bmf_module_sdk
)
