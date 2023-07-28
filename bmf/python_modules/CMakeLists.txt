
option(BMF_MODULE_ENABLE_BLUR_GPU "" ON)
option(BMF_MODULE_ENABLE_CPU_GPU_TRANS "" ON)
option(BMF_MODULE_ENABLE_CROP_GPU "" ON)
option(BMF_MODULE_ENABLE_FLIP_GPU "" ON)
option(BMF_MODULE_ENABLE_GAMMA_GPU "" ON)
option(BMF_MODULE_ENABLE_ROTATE_GPU "" ON)
option(BMF_MODULE_ENABLE_SCALE_GPU "" ON)

function(copy_python_builtin_module)
    file(GLOB files ${ARGV0}/*)
    file(COPY ${files} DESTINATION ${BMF_ASSEMBLE_ROOT}/bmf/python_modules/Module_${ARGV0})
endfunction()


if(BMF_MODULE_ENABLE_BLUR_GPU)
    copy_python_builtin_module(blur_gpu)
endif()

if(BMF_MODULE_ENABLE_CPU_GPU_TRANS)
    copy_python_builtin_module(cpu_gpu_trans_module)
endif()

if(BMF_MODULE_ENABLE_CROP_GPU)
    copy_python_builtin_module(crop_gpu)
endif()

if(BMF_MODULE_ENABLE_FLIP_GPU)
    copy_python_builtin_module(flip_gpu)
endif()

if(BMF_MODULE_ENABLE_GAMMA_GPU)
    copy_python_builtin_module(gamma_gpu)
endif()

if(BMF_MODULE_ENABLE_ROTATE_GPU)
    copy_python_builtin_module(rotate_gpu)
endif()

if(BMF_MODULE_ENABLE_SCALE_GPU)
    copy_python_builtin_module(scale_gpu)
endif()
