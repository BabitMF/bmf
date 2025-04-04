
message(STATUS "")
message(STATUS "================== BMF Summary =====================")
message(STATUS     "  CMAKE_SYSTEM_NAME       : ${CMAKE_SYSTEM_NAME}")
message(STATUS     "  CMAKE_CXX_COMPILER      : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS     "  CMAKE_EXPORT_COMMANDS   : ${CMAKE_EXPORT_COMPILE_COMMANDS}")
message(STATUS     "  BMF_BUILD_VERSION       : ${BMF_VERSION_MAJOR}.${BMF_VERSION_MINOR}.${BMF_VERSION_PATCH}")
message(STATUS     "  BMF_BUILD_COMMIT        : ${BMF_BUILD_COMMIT}")
message(STATUS "Dependences:")
message(STATUS     "  BREAKPAD_ENABLED        : ${BMF_ENABLE_BREAKPAD}")
message(STATUS     "  CUDA_ENABLED            : ${BMF_ENABLE_CUDA}")
if(BMF_ENABLE_CUDA)
    message(STATUS "      CUDA version        : ${CUDAToolkit_VERSION}")
    message(STATUS "      CUDA root directory : ${CUDAToolkit_LIBRARY_ROOT}")
    message(STATUS "      CUDA include path   : ${CUDAToolkit_INCLUDE_DIRS}")
endif()
message(STATUS     "  TORCH_ENABLED           : ${BMF_ENABLE_TORCH}")
message(STATUS     "  GLOG_ENABLED            : ${BMF_ENABLE_GLOG}")
message(STATUS     "  JNI_ENABLED             : ${BMF_ENABLE_JNI}")
message(STATUS     "  PYTHON_ENABLED          : ${BMF_ENABLE_PYTHON}")
if(BMF_ENABLE_PYTHON)
    message(STATUS "      PYTHON VERSION      : ${Python_VERSION}, ${PYTHON_MODULE_EXTENSION}")
endif()
message(STATUS     "  FFMPEG_ENABLED          : ${BMF_ENABLE_FFMPEG}")
message(STATUS     "  MOBILE_ENABLE           : ${BMF_ENABLE_MOBILE}")
message(STATUS     "  ENABLED_TEST            : ${BMF_ENABLE_TEST}")
message(STATUS     "  ENABLED_FUZZTEST        : ${BMF_ENABLE_FUZZTEST}")
message(STATUS     "  FUZZING_MODE            : ${FUZZTEST_ENABLE_FUZZING_MODE}")
