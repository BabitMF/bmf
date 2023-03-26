
set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
find_package(CUDA QUIET)

if(CUDA_FOUND AND HMP_ENABLE_CUDA)
    # cuda
    find_library(CUDA_LIB cuda 
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs lib/x64)
    add_library(cuda::cuda UNKNOWN IMPORTED)   
    set_target_properties(cuda::cuda PROPERTIES
        IMPORTED_LOCATION ${CUDA_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

    # cudart
    add_library(cuda::cudart INTERFACE IMPORTED GLOBAL)
    if(HMP_STATIC_LINK_CUDA)
        target_link_libraries(cuda::cudart INTERFACE "${CUDA_cudart_static_LIBRARY}")
        if (NOT WIN32)
            target_link_libraries(cuda::cudart INTERFACE rt dl)
        endif()
    else()
        set_property(
           TARGET cuda::cudart PROPERTY INTERFACE_LINK_LIBRARIES
           ${CUDA_LIBRARIES})
    endif()
    set_property(
        TARGET cuda::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${CUDA_INCLUDE_DIRS})

    # NPP
    add_library(cuda::npp INTERFACE IMPORTED GLOBAL)
    if(HMP_STATIC_LINK_CUDA)
        target_link_libraries(cuda::npp INTERFACE
             CUDA::nppc_static CUDA::nppicc_static CUDA::nppig_static)
    else()
        target_link_libraries(cuda::npp INTERFACE
             CUDA::nppc CUDA::nppicc CUDA::nppig_static)
    endif()

    if(HMP_CUDA_ARCH_FLAGS)
        cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${HMP_CUDA_ARCH_FLAGS})
    else()
        cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
    endif()
    list(APPEND CUDA_NVCC_FLAGS "--extended-lambda --expt-relaxed-constexpr ${CUDA_ARCH_FLAGS}")
elseif(HMP_ENABLE_CUDA)
    message("CUDA library not found, disable it\n")
    set(HMP_ENABLE_CUDA OFF)
endif()
