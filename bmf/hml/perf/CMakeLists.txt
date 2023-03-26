
file(GLOB PERF_SRCS *.cpp)

add_executable(hmp_perf_main ${PERF_SRCS})

target_link_libraries(hmp_perf_main 
    PRIVATE
        hmp benchmark benchmark_main)

if(HMP_ENABLE_OPENCV)
    target_link_libraries(hmp_perf_main 
        PRIVATE
            ${OpenCV_LIBS})
endif()

if(HMP_ENABLE_CUDA)
    target_include_directories(hmp_perf_main
        PRIVATE
            ${CUDA_INCLUDE_DIRS}
    )
endif()
