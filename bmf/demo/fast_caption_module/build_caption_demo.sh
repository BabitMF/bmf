#!/bin/bash
if [[ ! -d build ]];
then
    mkdir build
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release
cd build && make