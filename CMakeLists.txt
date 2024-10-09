cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Cross Compile
# cmake -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake

project(BMF CXX C)
### options
if (NOT APPLE)
    set(BMF_PYENV 3.7 CACHE STRING "Python version, ie. 3.6, 3.7, 3.8")
else ()
    set(BMF_PYENV)
endif ()
set(BMF_ASSEMBLE_ROOT ${CMAKE_BINARY_DIR}/output CACHE STRING "Directory to assymble BMF package")
set(BMF_BUILD_VERSION "" CACHE STRING "BMF version")
set(BMF_BUILD_COMMIT "" CACHE STRING "BMF commit")

option(BMF_LOCAL_DEPENDENCIES "Build dependencies locally" ON)
option(BMF_ENABLE_BREAKPAD "Enable build with breakpad support" OFF)
option(BMF_ENABLE_CUDA "Enable CUDA support" ON)
option(BMF_ENABLE_TORCH "Enable CUDA support" OFF)
option(BMF_ENABLE_PYTHON "Enable build with python support" ON)
option(BMF_ENABLE_GLOG "Enable build with glog support" OFF)
option(BMF_ENABLE_JNI "Enable build with JNI support" OFF)
option(BMF_ENABLE_FFMPEG "Enable build with ffmpeg support" ON)
option(BMF_ENABLE_MOBILE "Enable build for mobile platform" OFF)
option(BMF_ENABLE_TEST "Compile examples and tests" ON)
option(FUZZTEST_ENABLE_FUZZING_MODE "Enable fuzztest fuzzing mode" OFF)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compiler commands" OFF)

if (BMF_ENABLE_CUDA)
    # For FindCUDAToolkit support
    cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
endif()

if(ENABLE_BITCODE)
    set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE "bitcode")
    set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE "YES") 
else()
    set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")
endif()



### VERSION
string(REPLACE "." ";" VERSION_LIST ${BMF_BUILD_VERSION})
list(GET VERSION_LIST 0 BMF_VERSION_MAJOR)
list(GET VERSION_LIST 1 BMF_VERSION_MINOR)
list(GET VERSION_LIST 2 BMF_VERSION_PATCH)


### general settings
if(WIN32)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (${COVERAGE})
    set(CMAKE_COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage -O -g3")
else ()
    set(CMAKE_COVERAGE_FLAGS " ")
endif()

if(SANITIZERS MATCHES "address")
    set(CMAKE_SANITIZER_FLAGS "-fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-sanitize-recover=all")
elseif(SANITIZERS MATCHES "undefined")
    set(CMAKE_SANITIZER_FLAGS "-fno-sanitize-recover=all")
endif()

if(DEFINED CMAKE_SANITIZER_FLAGS)
    set(CMAKE_SANITIZER_FLAGS "-fsanitize=${SANITIZERS} ${CMAKE_SANITIZER_FLAGS}")
else()
    set(CMAKE_SANITIZER_FLAGS " ") 
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(SANITIZERS STREQUAL "")
        if ($ENV{SCRIPT_EXEC_MODE} MATCHES "x86")
            # runpath for bmf libs
            set(CMAKE_CXX_FLAGS "-D__STDC_FORMAT_MACROS -Wno-deprecated-declarations -Wl,-z,defs,-rpath,'$ORIGIN',--enable-new-dtags ${CMAKE_COVERAGE_FLAGS} ${CMAKE_SANITIZER_FLAGS} ${CMAKE_CXX_FLAGS}") # For python dist
        else()
            set(CMAKE_CXX_FLAGS "-D__STDC_FORMAT_MACROS -Wno-deprecated-declarations -Wl,-z,defs ${CMAKE_COVERAGE_FLAGS} ${CMAKE_SANITIZER_FLAGS} ${CMAKE_CXX_FLAGS}") # report unresolved symbols
        endif()
    else() # for sanitizers, do not report unresolved symbols (e.g. because ASAN relies on interposition)
        if ($ENV{SCRIPT_EXEC_MODE} MATCHES "x86")
            # runpath for bmf libs
            set(CMAKE_CXX_FLAGS "-D__STDC_FORMAT_MACROS -Wno-deprecated-declarations -Wl,-rpath,'$ORIGIN',--enable-new-dtags ${CMAKE_COVERAGE_FLAGS} ${CMAKE_SANITIZER_FLAGS} ${CMAKE_CXX_FLAGS}") # For python dist
        else()
            set(CMAKE_CXX_FLAGS "-D__STDC_FORMAT_MACROS -Wno-deprecated-declarations ${CMAKE_COVERAGE_FLAGS} ${CMAKE_SANITIZER_FLAGS} ${CMAKE_CXX_FLAGS}") # allow unresolved symbols
        endif()    
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_COVERAGE_FLAGS} ${CMAKE_SANITIZER_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld -pthread ${CMAKE_EXE_LINKER_FLAGS}")
    add_compile_definitions("BMF_ENABLE_FUZZ_TEST") # set macro for test executables with both gtest and fuzztest tests
    set(BMF_ENABLE_FUZZ_TEST "") # define cmake variable to optionally build test executables with fuzztest tests only
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4068: unknown pragma

    # C4576: a parenthesized type followed by an initializer list is a non-standard explicit type conversion syntax
    # Used in FFmpeg macros which assumes C.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor /wd4068 /wd4576")
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    if (BMF_ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
    endif()
endif()

if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -g")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Oz") # -O2 + -Oz

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
endif()


if(WIN32)
    # archive
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    # library
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    # binary
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BMF_ASSEMBLE_ROOT}/bmf/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BMF_ASSEMBLE_ROOT}/bmf/bin)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BMF_ASSEMBLE_ROOT}/bmf/bin)
endif()

if (EMSCRIPTEN)
    # Add some flags for wasm
    # -fPIC , USE_PTHREADS, SHARED_MEMORY for dynamic linking
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -s USE_PTHREADS=1 -s SHARED_MEMORY=1")
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
endif()

### dependencies
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS TRUE)  #build shared sdk by default
endif()

include(cmake/dependencies.cmake)

if(BMF_USE_MEDIACODEC)
    add_definitions("-DBMF_USE_MEDIACODEC")
endif()


### 
add_subdirectory(bmf)

## set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGSa} -Wl,--export-dynamic")


### print build configurations
include(cmake/summary.cmake)
