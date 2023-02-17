#!/bin/bash

# This script is only for cross-compilation for Android

# ARCH INDEXES (0 to disable, 1 to enable)
ARCH_ARM_V7A=0
ARCH_ARM64_V8A=1
ARCH_X86=2
ARCH_X86_64=3

# ENABLE ARCH
ENABLED_ARCHITECTURES=(0 1 0 0)

# ARCH ABIs
ANDROID_ABIS=("armeabi-v7a" "arm64-v8a" "x86" "x86_64")
ANDROID_PROCESSORS=("armv7-a" "aarch64" "i686" "x86_64")
ANDROID_ARCHS=("arm" "arm64" "x86" "x86_64")

# USING API LEVEL > 21
export ANDROID_API=21

# Default build type - Release
BUILD_TYPE="Release"

# Handle options
if [ $# > 0 ]
then
    # Clean up
    if [ "$1" = "clean" ]
    then
        rm -rf build_android
        rm -rf output
        exit
    fi

    # Debug type
    if [ "$1" = "debug" ]
    then
        BUILD_TYPE="Debug"
    fi
fi

if [[ -z ${ANDROID_NDK_ROOT} ]]
then
    echo "ANDROID_NDK_ROOT not defined"
    exit 1
fi


if [[ -z ${ANDROID_XCOMPILE_ROOTFS} ]]
then
    echo "ANDROID_XCOMPILE_ROOTFS not defined"
    exit 1
fi

DETECTED_NDK_VERSION=$(grep -Eo Revision.* ${ANDROID_NDK_ROOT}/source.properties | sed 's/Revision//g;s/=//g;s/ //g')

# Configuration
export SCRIPT_EXEC_MODE=android
export EXACT_PYTHON_TARGET=1
export PYTHON_VERSION=3.9

# Set the output directory
mkdir output

# Generate BMF version
source ./version.sh

for run_arch in {0..3}
do
    if [[ ${ENABLED_ARCHITECTURES[$run_arch]} -eq 1 ]]
    then
        export ANDROID_ARCH=${ANDROID_ARCHS[$run_arch]}
        export ANDROID_TARGET_PLATFORM=android-${ANDROID_API}

        export ANDROID_ROOTFS_PATH=${ANDROID_XCOMPILE_ROOTFS}/${ANDROID_ARCHS[$run_arch]}/usr
        export GLOG_ROOT_PATH=${ANDROID_ROOTFS_PATH}
        export FFMPEG_ROOT_PATH=${ANDROID_ROOTFS_PATH}

        export Python_LIBRARY=${ANDROID_ROOTFS_PATH}/lib/libpython${PYTHON_VERSION}.a
        export Python_INCLUDE_DIRS=${ANDROID_ROOTFS_PATH}/include/python${PYTHON_VERSION}

        echo -e "\nBuilding ${ANDROID_ABIS[$run_arch]} on API ${ANDROID_API}\n"
        echo $ANDROID_ROOTFS_PATH
        echo $Python_LIBRARY
        echo $Python_INCLUDE_DIR
        # Build
        mkdir -p build_android
        cd build_android
        cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DBMF_ENABLE_MOBILE=OFF \
            -DBMF_ENABLE_PYTHON=ON \
            -DBMF_ENABLE_GLOG=OFF \
            -DCMAKE_FIND_ROOT_PATH=${ANDROID_ROOTFS_PATH} \
            -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
            -DANDROID_STL=c++_shared \
            -DANDROID_ABI="${ANDROID_ABIS[$run_arch]}" \
            -DBMF_PYENV=${PYTHON_VERSION} \
            -DANDROID_PLATFORM=${ANDROID_TARGET_PLATFORM} \
            -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
            -DBMF_ENABLE_JNI=OFF \
            -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ..
        #cmake --build .
        #ninja 
        make -j16
        mkdir -p ../output/${ANDROID_ABIS[$run_arch]}
        rm -rf ../output/${ANDROID_ABIS[$run_arch]}/*
        cp -rf output/bmf output/hmp ../output/${ANDROID_ABIS[$run_arch]}/
        cd ..
        #rm -rf build_android
    fi
done

