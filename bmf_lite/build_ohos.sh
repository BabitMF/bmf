#!/bin/bash

# This script is only for cross-compilation for HarmonyOS

# USING OHOS_API >= 10
export OHOS_API=10

# Default build type - Release
BUILD_TYPE="Release"

# Set the output directory
rm -rf output
mkdir output

# Build
echo ${OHOS_NATIVE_ROOT}
mkdir -p build_ohos
cd build_ohos
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DANDROID_STL=c++_shared \
    -DCMAKE_TOOLCHAIN_FILE=${OHOS_NATIVE_ROOT}/build/cmake/ohos.toolchain.cmake \
    -DBMF_LITE_ENABLE_OPENGLTEXTUREBUFFER=ON \
    -DBMF_LITE_ENABLE_SUPER_RESOLUTION=ON

make -j16
mkdir -p ../output/${ANDROID_ABI}
rm -rf ../output/${ANDROID_ABI}/*
cp -r lib ../output/${ANDROID_ABI}
cd ..
