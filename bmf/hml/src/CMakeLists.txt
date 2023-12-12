

set(INC_DIR ${PROJECT_SOURCE_DIR}/include)

file(GLOB CPU_SRCS *.cpp *.h core/*.h core/*.cpp kernel/*.cpp imgproc/*.cpp)
file(GLOB CPU_HDRS ${INC_DIR}/hmp/*.h ${INC_DIR}/hmp/core/*.h kernel/*.h image/*.h)
file(GLOB CUDA_SRCS cuda/*.cpp)
file(GLOB CUDA_HDRS ${INC_DIR}/hmp/cuda/*.h)
file(GLOB CPU_KERN_SRCS
    kernel/*.h
    kernel/cpu/*.h
    kernel/cpu/*.cpp)
file(GLOB CUDA_KERN_SRCS kernel/cuda/*.h kernel/cuda/*.cu)

file(GLOB FFMPEG_SRCS ffmpeg/*.cpp ffmpeg/*.h)
file(GLOB FFMEPG_HDRS ${INC_DIR}/ffmepg/*.h)

file(GLOB OPENCV_KERN_SRCS kernel/cv2/*_cpu.cpp)
file(GLOB OPENCV_CUDA_KERN_SRCS kernel/cv2/*_cuda.cu kernel/cv2/*_cuda.cpp)

file(GLOB NPP_KERN_SRCS kernel/npp/*.h kernel/npp/*.cpp kernel/npp/*.cu)

file(GLOB TORCH_SRCS torch/*.cpp torch/*.h)
file(GLOB TORCH_HDRS ${INC_DIR}/torch/*.h)

file(GLOB DLPACK_SRCS dataexport/*.cpp dataexport/*.h)
file(GLOB DLPACK_HDRS ${INC_DIR}/dataexport/*.h)

if(HMP_ENABLE_MOBILE)
    list(FILTER CPU_SRCS EXCLUDE REGEX "image_seq.cpp")
    list(FILTER CPU_KERN_SRCS EXCLUDE REGEX "imgproc|binary_ops|unary_ops")
    list(FILTER CPU_SRCS EXCLUDE REGEX "hmp_capi")
endif()

set(HMP_SRCS ${CPU_SRCS} ${CPU_KERN_SRCS})
set(HMP_HDRS ${CPU_HDRS})

# FFMPEG
if(HMP_ENABLE_FFMPEG)
    list(APPEND HMP_SRCS ${FFMPEG_SRCS})
    list(APPEND HMP_HDRS ${FFMEPG_HDRS})
endif()

# OpenCV
if(HMP_ENABLE_OPENCV)
    list(APPEND HMP_SRCS ${OPENCV_KERN_SRCS})
endif()

# Torch
if(HMP_ENABLE_TORCH)
    list(APPEND HMP_SRCS ${TORCH_SRCS})
    list(APPEND HMP_HDRS ${TORCH_HDRS})
endif()

# DLPack
list(APPEND HMP_SRCS ${DLPACK_SRCS})
list(APPEND HMP_SRCS ${DLPACK_HDRS})

if(HMP_ENABLE_CUDA)
    enable_language(CUDA)
    if(HMP_ENABLE_OPENCV)
        list(APPEND HMP_SRCS ${OPENCV_CUDA_KERN_SRCS})
    endif()

    if(HMP_ENABLE_NPP)
        list(APPEND HMP_SRCS ${NPP_KERN_SRCS})
    endif()

    list(APPEND HMP_SRCS ${CUDA_SRCS} ${CUDA_KERN_SRCS})
    list(APPEND HMP_HDRS ${CUDA_HDRS})
    add_library(hmp ${HMP_SRCS} ${HMP_HDRS})
else()
    add_library(hmp ${HMP_SRCS} ${HMP_HDRS})
endif()

if(WIN32)
    set_target_properties(hmp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib
    )
endif()

target_include_directories(hmp
    PUBLIC
        $<BUILD_INTERFACE:${INC_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PRIVATE
        ${INC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

if (HMP_LOCAL_DEPENDENCIES)
    target_include_directories(hmp
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/spdlog/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/fmt/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/dlpack/include>
    )
endif()

target_link_libraries(hmp
    PUBLIC
        ${HMP_CORE_PUB_DEPS}
    PRIVATE
        ${HMP_CORE_PRI_DEPS}
    )

if(ANDROID)
    target_compile_options(hmp PRIVATE -std=c++11)
    target_link_libraries(hmp PRIVATE ${ANDROID_STL} log)
endif()

target_compile_options(hmp
    PRIVATE
        -D_FILE_OFFSET_BITS=64
    )

set_target_properties(hmp PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
)

if(BUILD_SHARED_LIBS)
    target_compile_options(hmp
        PRIVATE
            -DHMP_BUILD_SHARED
        )
endif()


if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Android|iOS|Darwin")
    set_target_properties(hmp PROPERTIES
        VERSION "${HMP_VERSION_MAJOR}.${HMP_VERSION_MINOR}.${HMP_VERSION_PATCH}"
        SOVERSION "${HMP_VERSION_MAJOR}"
    )
endif()

add_subdirectory(tests)
