#!/bin/bash

# This script is only for cross-compilation for Android

# ARCH ABIs
ANDROID_ABIS=("armeabi-v7a" "arm64-v8a" "x86" "x86_64")
ANDROID_ABI="arm64-v8a"
# USING API LEVEL > 21
export ANDROID_API=21

# Default build type - Release
BUILD_TYPE="Release"


# Set the output directory
rm -rf output
mkdir output


# Build
echo ${ANDROID_NDK_ROOT}
mkdir -p build_android
cd build_android
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DANDROID_STL=c++_shared \
    -DANDROID_ABI="arm64-v8a" \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
    -DBMF_LITE_ENABLE_OPENGLTEXTUREBUFFER=ON \
    -DBMF_LITE_ENABLE_CPUMEMORYBUFFER=ON \
    -DBMF_LITE_ENABLE_SUPER_RESOLUTION=ON \
    -DBMF_LITE_ENABLE_DENOISE=ON \
    -DBMF_LITE_ENABLE_TEX_GEN_PIC=ON

    
make -j16
mkdir -p ../output/${ANDROID_ABI}
rm -rf ../output/${ANDROID_ABI}/*
cp -r lib ../output/${ANDROID_ABI}
cd ..
