#!/bin/bash

set -exuo pipefail

cd 3rd_party

if [ -d breakpad ]
then
    rm -rf breakpad
fi

#install depot_tools
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH=$(pwd)/depot_tools:$PATH

mkdir breakpad && cd breakpad
fetch breakpad
INSTALL_PREFIX=$(pwd)
cd src
git checkout v2022.07.12 #latest tag is v2023.01.27, higher version of libelf that not installed locally is required

#install breakpad
./configure --prefix=${INSTALL_PREFIX}
make -j $(nproc)
make install