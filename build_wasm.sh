#!/bin/bash

# Default build type - Debug
# The wasm support is still under constructing.
BUILD_TYPE="Debug"


mkdir -p output
LOCAL_BUILD=1
git submodule update --init --recursive

# need to edit some files
backward_hpp=bmf/hmp/third_party/backward/backward.hpp
backward_config=bmf/hmp/third_party/backward/BackwardConfig.cmake
sed -i 's/#ifdef __CLANG_UNWIND_H/#if defined __CLANG_UNWIND_H \&\& !defined EMSCRIPTEN/' $backward_hpp
if ! grep -q "EMSCRIPTEN" $backward_config; then
    # if we have never edited before..
    sed -i '/if (STACK_DETAILS_BFD)/i if (NOT EMSCRIPTEN)' $backward_config
    sed -i '/if (STACK_DETAILS_BFD)/{N; :a; /endif/!{N; ba}; s/endif()/&\nendif()/;}' $backward_config
fi


(cd 3rd_party/dlpack && emcmake cmake . -DCMAKE_INSTALL_PREFIX=="$PWD/output" && emmake make && make install)

# Generate BMF version
source ./version.sh

source ../emsdk/emsdk_env.sh
mkdir -p build && cd build
# TODO:You may need specify a path to your ffmpeg library.
export FFMPEG_ROOT_PATH=/path/to/your/ffmpeg/lib

emcmake cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCOVERAGE=${COVERAGE_OPTION} \
    -DBMF_LOCAL_DEPENDENCIES=${LOCAL_BUILD} \
    -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
    -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} \
    -DBMF_ENABLE_PYTHON=OFF \
    -DBMF_ENABLE_TEST=OFF \
    -DBMF_ENABLE_FFMPEG=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBMF_ENABLE_CUDA=OFF
    
emmake make -j4