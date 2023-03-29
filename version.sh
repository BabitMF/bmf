#!/bin/bash

BMF_BUILD_VERSION=$(python setup.py --version)

if echo "Using git: " && git --version
then
    BMF_BUILD_COMMIT=$(git rev-parse --short HEAD)
else
    BMF_BUILD_COMMIT="0"   # No git, use default 0 as build version
fi

echo "BMF Version: ${BMF_BUILD_VERSION}"
echo "BMF Commit: ${BMF_BUILD_COMMIT}"