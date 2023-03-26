#!/bin/bash

# Default build type - Release
BUILD_TYPE="Release"

# Retrieve submodules
#git submodule update --init --recursive    # unable to do so in scm
cp -a -r /usr/local/glog/. 3rd_party/glog
cp -a -r /usr/local/googletest/. 3rd_party/gtest

# Generate BMF version
source ./version.sh

# Build Arm
export HOME=/usr/local
mkdir -p build_aarch64
cd build_aarch64
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake \
    -DBMF_BOOST_STATIC_LIBS=ON \
    -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
    -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ..
make -j8

# Reduce output size
rm -rf output/example
rm -rf output/test

# Output
mv output ../