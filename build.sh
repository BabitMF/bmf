#!/bin/bash

# BMF compilation script
#
# To build for x86 (default):
#   ./build.sh
#
# To clean up:
#   ./build.sh clean
#
# For debug build:
#   ./build.sh debug
#
# For debug build and generating coverage report:
#   ./build.sh with_cov
#
# To build with non local dependencies:
#   ./build.sh non_local
#

# Default build type - Release
BUILD_TYPE="Release"

# For SCM compilation only:
#
# To compile for x86 (multiple Python versions):
#   export SCRIPT_EXEC_MODE=x86
#
# Note: x86 package will include Python sdist for Python 3.6 - 3.9 (located in
# dist folder). The other folders are catered for SCM installation method.

COVERAGE_OPTION=0
LOCAL_BUILD=1

# Handle options
if [ $# -gt 0 ]
then
    # Clean up
    if [ "$1" = "clean" ]
    then
        rm -rf build
        exit
    fi

    # Debug type
    if [ "$1" = "debug" ]
    then
        BUILD_TYPE="Debug"
    fi

    # Debug type with coverage option
    if [ "$1" = "with_cov" ]
    then
        BUILD_TYPE="Debug"
        COVERAGE_OPTION=1
    fi

    # Debug type with coverage option
    if [ "$1" = "non_local" ]
    then
        LOCAL_BUILD=0
    fi
    if [ "$1" = "disable_cuda" ]
    then
        CUDA_ENABLE=OFF
    fi
fi

mkdir -p output

if [ $LOCAL_BUILD -ne 0 ]
then
  git submodule update --init --recursive
fi

(cd 3rd_party/dlpack && cmake . && make && make install)

if [ ! -d "3rd_party/breakpad" ]
then
    (cd 3rd_party/ && wget https://github.com/BabitMF/bmf/releases/download/files/breakpad.tar.xz && tar xvf breakpad.tar.xz)
fi

# Generate BMF version
source ./version.sh

cmake_args=""
if [[ ! -z "${CMAKE_ARGS}" ]]
then
    cmake_args="${cmake_args} ${CMAKE_ARGS}"
fi

# Handle SCM compilation for x86 multiple Python versions
if [ "$SCRIPT_EXEC_MODE" == "x86" ]
then
    export EXACT_PYTHON_TARGET=1

    # Create an output directory at repo root for packaging
    mkdir -p output/dist
    mkdir -p output/3rd_party
    cp bmf/c_modules/meta/BUILTIN_CONFIG.json bmf

    # Build x86
    python_versions=(3.6 3.7 3.8 3.9)
    python_names=(36 37 38 39)
    for i in "${!python_versions[@]}"
    do
        mkdir -p build && cd build
        cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DBMF_PYENV="${python_versions[$i]}" \
            -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
            -DBMF_ENABLE_TEST=OFF \
            -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ${cmake_args} ..
        make -j$(nproc)

        # Transfer to output directory to package
        mkdir output/bmf/3rd_party

        # Store the pre-compiled dependencies
        cp /usr/lib/x86_64-linux-gnu/libgflags.so.2.2 output/bmf/3rd_party
        cp /usr/lib/x86_64-linux-gnu/libglog.so.0 output/bmf/3rd_party
        cp /usr/lib/x86_64-linux-gnu/libunwind.so.8 output/bmf/3rd_party

        # Patch rpath for dependencies
        cd output/bmf/3rd_party
        for libfile in *.so*
        do
            if [ -f "$libfile" ]; then 
                patchelf --set-rpath '$ORIGIN' "$libfile"
                echo "Patch $libfile"
            fi
        done
        cd ../../..

        cp -r output/bmf/lib/. output/bmf/3rd_party
        cp /usr/local/lib/libpython"${python_versions[$i]}"* output/bmf/3rd_party
        mv output/bmf/3rd_party ../bmf/lib
        cp -r output/bmf/bin ../bmf/bin
        cp -r /usr/local/boost_1_68_0/boost output/bmf/include/
        cp -r /usr/include/glog output/bmf/include/
        cp -r ../3rd_party/json/include/. output/bmf/include/
        cp -r output/bmf/include ../bmf/include

        # Package for specific python version
        cd ..
        /usr/bin/python3.7 setup.py bdist_wheel
        cd dist
        for file in *.whl
        do
            echo "$file"
            mv "$file" ../output/dist/"${file%-py3-none-any.whl}-cp${python_names[$i]}-none-linux_x86_64.whl"
        done
        cd ..

        # Only include example and test for the default package
        if [ "${python_versions[$i]}" == "3.7" ]
        then
            mv build/output/bmf output
            mv build/output/example output/example
            mv build/output/test output/test
        fi

        rm -rf build
        rm -rf dist
        rm -rf bmf/lib
        rm -rf bmf/bin
        rm -rf bmf/include
    done

    mkdir -p output/3rd_party/lib
    cp -a -r 3rd_party/json output/3rd_party/
    # glog
    cp /usr/lib/x86_64-linux-gnu/libglog.so.* output/3rd_party/lib
    # boost
    cp /usr/local/boost_1_68_0/stage/lib/libboost_numpy37.so.1.68.0* output/3rd_party/lib
    cp /usr/local/boost_1_68_0/stage/lib/libboost_system.so.1.68.0* output/3rd_party/lib
    cp /usr/local/boost_1_68_0/stage/lib/libboost_python37.so.1.68.0* output/3rd_party/lib
    cp /usr/local/boost_1_68_0/stage/lib/libboost_filesystem.so.1.68.0* output/3rd_party/lib
    # others
    cp /usr/lib/x86_64-linux-gnu/libgflags.so.2.2* output/3rd_party/lib
    cp /usr/lib/x86_64-linux-gnu/libunwind.so.8* output/3rd_party/lib
    cp /usr/lib/x86_64-linux-gnu/libpython3.7m.so.1.0* output/3rd_party/lib

# Default build mode
else
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DBMF_PYENV=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))") \
        -DCOVERAGE=${COVERAGE_OPTION} \
        -DBMF_LOCAL_DEPENDENCIES=${LOCAL_BUILD} \
        -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
        -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ${cmake_args} ..
    make -j$(nproc)

    # Transfer to output directory
    cd ..
    cp -a -r build/output/. output
    rm -rf output/bmf/files
    rm -rf output/example/files

    # Breakpad
    if [[ "${cmake_args}" =~ "-DBMF_ENABLE_BREAKPAD=ON" ]]
    then
        python3 create_symbols.py -b 3rd_party/breakpad/bin -s output/bmf/lib -d ./symbols
        if [ -d "symbols" ]
        then
            cp -r symbols output/
        fi
    fi
fi
