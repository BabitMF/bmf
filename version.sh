#!/bin/bash
# Check if BMF_VERSION_OVERRIDE is set and use it if available
if [ -n "${BMF_VERSION_OVERRIDE:-}" ]; then
    BMF_BUILD_VERSION="${BMF_VERSION_OVERRIDE}"
else
    # Otherwise use the existing logic to get version from setup.py
    if [[ $OS == *Windows* ]]; 
    then
        BMF_BUILD_VERSION=$(python setup.py --version)
    else
        if [ "$(uname -s)" = "Darwin" ]; then
            BMF_BUILD_VERSION=$(awk -F\" '/package_version=/ {print $2}' setup.py)
        else
            BMF_BUILD_VERSION=$(cat setup.py | grep "package_version=" | grep -oP '"\K[0-9.]+')
        fi
    fi
fi

if echo "Using git: " && git --version
then
    BMF_BUILD_COMMIT=$(git rev-parse --short HEAD)
else
    BMF_BUILD_COMMIT="0"   # No git, use default 0 as build version
fi

echo "BMF Version: ${BMF_BUILD_VERSION}"
echo "BMF Commit: ${BMF_BUILD_COMMIT}"
