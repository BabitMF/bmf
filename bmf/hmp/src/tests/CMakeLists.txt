
set(TEST_SRCS 
    test_image.cpp
    test_scalar.cpp 
    test_allocator.cpp
    test_ref_ptr.cpp
    test_tensor_options.cpp
    )

set(TEST_CUDA_SRCS)

if(HMP_ENABLE_CUDA)
    list(APPEND TEST_SRCS ${TEST_CUDA_SRCS})
endif()

add_executable(test_hmp ${TEST_SRCS})

target_link_libraries(test_hmp PRIVATE gtest gtest_main hmp)

if(HMP_ENABLE_FFMPEG)
    target_link_libraries(test_hmp PRIVATE ffmpeg::avcodec ffmpeg::avutil )
    if(HMP_ENABLE_CUDA)
        target_link_libraries(test_hmp PRIVATE cuda::cuda)
    endif()
endif()

if(ANDROID)
    target_link_libraries(test_hmp PRIVATE ${ANDROID_STL} log)
endif()

add_test(NAME HMP_CORE_CPP_TESTS COMMAND $<TARGET_FILE:test_hmp>)