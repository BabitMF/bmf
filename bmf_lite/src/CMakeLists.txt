if(ANDROID)
    # file(GLOB_RECURSE Src ./*.cpp)
    include_directories(./)
    include_directories(./algorithm/modules/operators)
    include_directories(./algorithm/modules/operators/utils)
    include_directories(../api/cpp/)

    list(APPEND Src ./algorithm/algorithm_interface.cpp
                    ./algorithm/bmf_algorithm_impl.cpp
                    ./algorithm/bmf_video_frame.cpp
                    ./algorithm/bmf_algorithm.cpp
                    ./algorithm/bmf_contrib_algorithm.cpp
                    ./common/bmf_param.cpp
                    ./media/video_buffer/video_buffer_allocator.cpp
                    ./media/video_buffer/video_buffer_pool.cpp
                    ./media/video_buffer/hardware_device_context.cpp
                    ./media/video_buffer/raw_video_buffer.cpp
                    ./media/video_buffer/video_buffer.cpp
                    ./algorithm/modules/operators/utils/SHA256.cpp)

    if(BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER)
        add_compile_definitions(BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER)
        list(APPEND Src ./media/video_buffer/gl_texture_video_buffer/egl_hardware_device_context.cpp
                        ./media/video_buffer/gl_texture_video_buffer/gl_texture_video_buffer_allocator.cpp
                        ./media/video_buffer/gl_texture_video_buffer/gl_texture_video_buffer.cpp)
    endif()

    if(BMF_LITE_ENABLE_CPUMEMORYBUFFER)
        include_directories(./media/video_buffer/memory_video_buffer)
        list(APPEND Src ./media/video_buffer/memory_video_buffer/cpu_memory_video_buffer.cpp
                        ./media/video_buffer/memory_video_buffer/cpu_memory_video_buffer_allocator.cpp)
        add_compile_definitions(BMF_LITE_ENABLE_CPUMEMORYBUFFER)
    endif()

    if(BMF_LITE_ENABLE_SUPER_RESOLUTION)
        include_directories(algorithm/modules/super_resolution)
        add_compile_definitions(BMF_LITE_ENABLE_SUPER_RESOLUTION)
        list(APPEND Src ./algorithm/modules/super_resolution/super_resolution_algorithm.cpp
                        ./algorithm/modules/operators/opengl/sr/sr.cpp)
        add_compile_definitions(BMF_LITE_ENABLE_SUPER_RESOLUTION)
    endif()

    if(BMF_LITE_ENABLE_DENOISE)
        include_directories(algorithm/modules/denoise)
        add_compile_definitions(BMF_LITE_ENABLE_DENOISE)
        list(APPEND Src ./algorithm/modules/denoise/denoise_algorithm.cpp
                        ./algorithm/modules/operators/opengl/denoise/denoise.cpp)
        add_compile_definitions(BMF_LITE_ENABLE_DENOISE)
    endif()

    if(BMF_LITE_ENABLE_TEX_GEN_PIC)
        add_compile_definitions(BMF_LITE_ENABLE_TEX_GEN_PIC)
        include_directories(./algorithm/contrib_modules/QNNControlNet)
        include_directories(./algorithm/contrib_modules/QNNControlNet/QNN)
        include_directories(./algorithm/contrib_modules/QNNControlNet/scheduler)
        include_directories(./algorithm/contrib_modules/QNNControlNet/src)
        include_directories(./algorithm/contrib_modules/QNNControlNet/utils)
        include_directories(./media/video_buffer/transform)
        include_directories(./algorithm/modules/operators/opengl/resize)
        list(APPEND Src ./algorithm/contrib_modules/QnnControlNet/scheduler/scheduler_dpmpp_2m.cpp
                        ./algorithm/contrib_modules/QnnControlNet/inference/QnnModel.cpp
                        ./algorithm/contrib_modules/QnnControlNet/inference/QnnRuntime.cpp
                        ./algorithm/contrib_modules/QnnControlNet/inference/QnnTensorData.cpp
                        ./algorithm/contrib_modules/QnnControlNet/utils/time_all.cpp
                        ./algorithm/contrib_modules/QnnControlNet/QnnControlNet_algorithm.cpp
                        ./algorithm/contrib_modules/QnnControlNet/QnnControlNetPipeLine.cpp
                        ./media/video_buffer/transform/gl_texture_transformer.cpp
                        ./algorithm/modules/operators/opengl/resize/resize.cpp)
    endif()

    add_library(bmf_lite ${Src})
    target_link_libraries(bmf_lite PUBLIC EGL log jnigraphics)

elseif(APPLE)
    include_directories(./)
    include_directories(../api/cpp/)
    include_directories(algorithm/modules/operators)

    set(Src)
    set(METAL_SRCS)

    list(APPEND Src algorithm/algorithm_interface.cpp
                    algorithm/bmf_algorithm_impl.cpp
                    algorithm/bmf_video_frame.cpp
                    algorithm/bmf_algorithm.cpp
                    algorithm/bmf_contrib_algorithm.cpp
                    common/bmf_param.cpp
                    media/video_buffer/video_buffer_allocator.cpp
                    media/video_buffer/video_buffer_pool.cpp
                    media/video_buffer/hardware_device_context.cpp
                    media/video_buffer/raw_video_buffer.cpp
                    media/video_buffer/video_buffer.cpp)

    if(BMF_LITE_ENABLE_CVPIXELBUFFER)
        include_directories(media/video_buffer/cvpixelbuffer_video_buffer)
        list(APPEND Src media/video_buffer/cvpixelbuffer_video_buffer/cvpixelbuffer_video_buffer.mm
                        media/video_buffer/cvpixelbuffer_video_buffer/cvpixelbuffer_video_buffer_allocator.mm)
        add_compile_definitions(BMF_LITE_ENABLE_CVPIXELBUFFER)
    endif()

    if(BMF_LITE_ENABLE_METALBUFFER)
        include_directories(media/video_buffer/metal_texture_video_buffer)
        list(APPEND Src media/video_buffer/metal_texture_video_buffer/metal_texture_video_buffer.mm
                        media/video_buffer/metal_texture_video_buffer/mtl_device_context.mm
                        media/video_buffer/metal_texture_video_buffer/multi_metal_texture_video_buffer_allocator.mm
                        media/video_buffer/metal_texture_video_buffer/metal_texture_video_buffer_allocator.mm
                        media/video_buffer/metal_texture_video_buffer/multi_metal_texture_video_buffer.mm)
        add_compile_definitions(BMF_LITE_ENABLE_METALBUFFER)
    endif()

    if(BMF_LITE_ENABLE_CPUMEMORYBUFFER)
        include_directories(media/video_buffer/memory_video_buffer)
        list(APPEND Src media/video_buffer/memory_video_buffer/cpu_memory_video_buffer.cpp
                        media/video_buffer/memory_video_buffer/cpu_memory_video_buffer_allocator.cpp)
        add_compile_definitions(BMF_LITE_ENABLE_CPUMEMORYBUFFER)
    endif()

    if(BMF_LITE_ENABLE_BUFFER_TRANSFORM)
        include_directories(media/video_buffer/transform)
        list(APPEND Src media/video_buffer/transform/cvpixelbuffer_transformer.mm)
    endif()

    if(BMF_LITE_ENABLE_SUPER_RESOLUTION)
        include_directories(algorithm/modules/super_resolution)
        list(APPEND Src algorithm/modules/super_resolution/super_resolution_algorithm.mm
                        algorithm/modules/operators/metal/sr/sr.mm)
        list(APPEND METAL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/modules/operators/metal/sr/sr.metal)
        add_compile_definitions(BMF_LITE_ENABLE_SUPER_RESOLUTION)
    endif()

    if(BMF_LITE_ENABLE_DENOISE)
        include_directories(algorithm/modules/denoise)
        list(APPEND Src algorithm/modules/denoise/denoise_algorithm.mm
                        algorithm/modules/operators/metal/denoise/denoise.mm)
        list(APPEND METAL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/modules/operators/metal/denoise/denoise.metal)
        add_compile_definitions(BMF_LITE_ENABLE_DENOISE)
    endif()

    if(BMF_LITE_ENABLE_CANNY)
        include_directories(algorithm/contrib_modules/canny)
        list(APPEND Src algorithm/contrib_modules/canny/canny_algorithm.mm)
        add_compile_definitions(BMF_LITE_ENABLE_CANNY)
    endif()

    list(REMOVE_DUPLICATES Src)
    message(STATUS "Src:${Src}")
    add_library(bmf_lite ${Src})
    target_link_libraries(bmf_lite PUBLIC "-framework AVFoundation" "-framework Metal" "-framework MetalPerformanceShaders")

    list(REMOVE_DUPLICATES METAL_SRCS)
    message(STATUS "METAL SRC:${METAL_SRCS}")

    set(METAL_COMMANDS)
    set(METAL_AIRS)
    foreach(F ${METAL_SRCS})
        list(APPEND METAL_COMMANDS COMMAND xcrun -sdk iphoneos metal -std=ios-metal2.0 -mios-version-min=11.0 -c ${F} -o ${F}.air)
        list(APPEND METAL_AIRS ${F}.air)
    endforeach()

    add_custom_command(TARGET bmf_lite POST_BUILD
        ${METAL_COMMANDS}
        COMMAND xcrun -sdk iphoneos metal -std=ios-metal2.0 -mios-version-min=11.0 ${METAL_AIRS} -o ${CMAKE_CURRENT_BINARY_DIR}/bmf_lite.metallib
    )

    set_target_properties(bmf_lite PROPERTIES RESOURCE ${CMAKE_CURRENT_BINARY_DIR}/hydra.metallib)

elseif(OHOS)
    file(GLOB_RECURSE Src ./algorithm/modules/*.cpp)

    list(APPEND Src algorithm/algorithm_interface.cpp
                    algorithm/bmf_algorithm_impl.cpp
                    algorithm/bmf_video_frame.cpp
                    algorithm/bmf_algorithm.cpp
                    algorithm/bmf_contrib_algorithm.cpp
                    common/bmf_param.cpp
                    media/video_buffer/video_buffer_allocator.cpp
                    media/video_buffer/video_buffer_pool.cpp
                    media/video_buffer/hardware_device_context.cpp
                    media/video_buffer/raw_video_buffer.cpp
                    media/video_buffer/video_buffer.cpp)

    include_directories(./)
    include_directories(../api/cpp/)
    include_directories(./algorithm/modules/operators)
    include_directories(./algorithm/modules/denoise)
    include_directories(./algorithm/modules/super_resolution)
    include_directories(./media/video_buffer/transform)

    if(BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER)
        list(APPEND Src media/video_buffer/gl_texture_video_buffer/egl_hardware_device_context.cpp
                        media/video_buffer/gl_texture_video_buffer/gl_texture_video_buffer.cpp
                        media/video_buffer/gl_texture_video_buffer/gl_texture_video_buffer_allocator.cpp)
        add_compile_definitions(BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER)
    endif()

    if(BMF_LITE_ENABLE_SUPER_RESOLUTION)
        add_compile_definitions(BMF_LITE_ENABLE_SUPER_RESOLUTION)
    endif()

    if(BMF_LITE_ENABLE_DENOISE)
        add_compile_definitions(BMF_LITE_ENABLE_DENOISE)
    endif()

    add_library(bmf_lite ${Src})
    target_link_libraries(bmf_lite PUBLIC EGL hilog_ndk.z GLESv3)

else()
    message(FATAL "Unsupport plateformat")
endif()