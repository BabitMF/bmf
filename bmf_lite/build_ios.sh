#! /usr/bin/env bash



set -euxo pipefail

echo "bmf_ios_build"
TARGETCODEPATH=$PWD
cd $TARGETCODEPATH
BMF_BUILD_VERSION="1.0.1"
S_ENABLE=1
S_DISABLE=0
BUILD_CONFIG="Release"

VCB_DEBUG=${VCB_DEBUG:-${S_DISABLE}}
VCB_BITCODE=${VCB_BITCODE:-${S_DISABLE}}
VCB_VERSION=${VCB_VERSION:-"1.0.0"}

if [ $VCB_DEBUG -eq $S_ENABLE ];then
    echo "build debug"
    BUILD_CONFIG="Debug"
fi

# source ./version.sh

rm -rf build_ios

mkdir -p build_ios
cd build_ios
CURRENT_DIR_PATH=`pwd`
echo "current path $CURRENT_DIR_PATH"

cmake -GXcode \
      -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR_PATH}/../iosoutputstatic \
      -DCMAKE_TOOLCHAIN_FILE=../cmake/ios.toolchain.cmake \
      -DPLATFORM=OSIPHONEALLCOMBINED 	\
      -DBMF_LITE_ENABLE_METALBUFFER=ON \
      -DBMF_LITE_ENABLE_CVPIXELBUFFER=ON \
      -DBMF_LITE_ENABLE_BUFFER_TRANSFORM=ON \
      -DBMF_LITE_ENABLE_SUPER_RESOLUTION=ON \
      -DBMF_LITE_ENABLE_DENOISE=ON \
      -DBMF_LITE_ENABLE_CANNY=ON \
      -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
      -DENABLE_BITCODE=OFF  	\
      -DDEPLOYMENT_TARGET=9.0  \
      -DBUILD_SHARED_LIBS=OFF 	\
      -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO  \
      -DCMAKE_SYSTEM_NAME=iOS  \
      -DCMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC=YES \
      ..

cd $TARGETCODEPATH

pwd

PROJECT_PATH="build_ios"

INSTALL_PREFIX="build_ios/output/ios"
POD_SPEC_NAME="hmp"
rm -rf $INSTALL_PREFIX

cmake --build $PROJECT_PATH --target bmf_lite --config $BUILD_CONFIG -- -sdk iphoneos VALID_ARCHS="armv7 arm64"
