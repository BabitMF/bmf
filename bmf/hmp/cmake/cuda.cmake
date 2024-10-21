
find_package(CUDAToolkit 11 COMPONENTS cudart cuda_driver NPP)
if(CUDAToolkit_FOUND AND HMP_ENABLE_CUDA)
    # cuda
    add_library(cuda::cuda ALIAS CUDA::cuda_driver)

    # cudart
    if(HMP_STATIC_LINK_CUDA)
        add_library(cuda::cudart ALIAS CUDA::cudart_static)
    else()
        add_library(cuda::cudart ALIAS CUDA::cudart)
    endif()

    # NPP
    add_library(cuda::npp INTERFACE IMPORTED GLOBAL)
    if(HMP_STATIC_LINK_CUDA)
        target_link_libraries(cuda::npp INTERFACE
             CUDA::nppc_static CUDA::nppicc_static CUDA::nppig_static)
    else()
        target_link_libraries(cuda::npp INTERFACE
             CUDA::nppc CUDA::nppicc CUDA::nppig)
    endif()

    if(HMP_CUDA_ARCH_FLAGS)
        set(CMAKE_CUDA_ARCHITECTURES ${HMP_CUDA_ARCH_FLAGS})
    else()
        set(CMAKE_CUDA_ARCHITECTURES all)
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
elseif(HMP_ENABLE_CUDA)
    message("CUDA library not found, disable it\n")
    set(HMP_ENABLE_CUDA OFF)
endif()
