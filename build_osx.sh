#!/bin/bash

# This script is for Mac OSX cross-compilation on Linux as well as local compilation on Mac
# For cross-compilation, precompiled tools, libraries and paths are required from https://
# For local compilation on Mac, refer to https://

# Default build type - Release
BUILD_TYPE="Release"

# Handle options
if [ $# > 0 ]
then
    # Clean up
    if [ "$1" = "clean" ]
    then
        rm -rf build_osx
        exit
    fi

    # Debug type
    if [ "$1" = "debug" ]
    then
        BUILD_TYPE="Debug"
    fi
fi

# Generate BMF version
source ./version.sh

if [[ "$OSTYPE" == "darwin"* ]]
then
    BMF_PYVER="3.7"
    if [[ -z "${BMF_PYTHON_VERSION}" ]]
    then
        echo "BMF_PYTHON_VERSION is not set, using default Python 3.7"
    else
        echo "Compiling for Python ${BMF_PYTHON_VERSION}"
        BMF_PYVER="${BMF_PYTHON_VERSION}"
    fi

    mkdir -p build_osx
    cd build_osx
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DBMF_PYENV=${BMF_PYVER} \
        -DCOVERAGE=${COVERAGE_OPTION} \
        -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
        -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ..
    make -j8

    cd output/bmf/lib
    for file in *.dylib
    do
        # Copy all .dylib to .so
        cp "$file" "${file/.dylib/.so}"
    done

    # Transfer to output directory
    cd ../../../..
    cp -r build_osx/output/. output
else
    # Configuration
    export SCRIPT_EXEC_MODE=osx
    export EXACT_PYTHON_TARGET=1
    export OSX_XCOMPILE_ROOT=/usr/local
    export OSX_XCOMPILE_TARGET=x86_64-apple-darwin19
    export OSX_XCOMPILE_TOOLS=$OSX_XCOMPILE_ROOT/osxcross
    export OSX_XCOMPILE_TOOL_PREFIX=$OSX_XCOMPILE_TOOLS/target/bin/x86_64-apple-darwin19
    export INSTALL_NAME_TOOL_PATH=$OSX_XCOMPILE_TOOLS/target/bin/x86_64-apple-darwin19-install_name_tool

    # Set the output directory
    mkdir output
    mkdir output/dist
    cp bmf/engine/c_engine/BUILTIN_CONFIG.json bmf

    # Compile for all listed python versions
    python_versions=(3.6 3.7 3.8 3.9)
    python_names=(36 37 38 39)
    for i in "${!python_versions[@]}"
    do
        if [ "${python_versions[$i]}" = "3.6" ] || [ "${python_versions[$i]}" = "3.7" ]
        then
            export Python_LIBRARY=$OSX_XCOMPILE_ROOT/osx-rootfs/usr/lib/libpython"${python_versions[$i]}"m.dylib
            export Python_INCLUDE_DIR=$OSX_XCOMPILE_ROOT/osx-rootfs/usr/include/python"${python_versions[$i]}"m
        else
            export Python_LIBRARY=$OSX_XCOMPILE_ROOT/osx-rootfs/usr/lib/libpython"${python_versions[$i]}".dylib
            export Python_INCLUDE_DIR=$OSX_XCOMPILE_ROOT/osx-rootfs/usr/include/python"${python_versions[$i]}"
        fi

        mkdir -p build_osx
        cd build_osx
        cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DCMAKE_TOOLCHAIN_FILE=../cmake/osx-toolchain.cmake \
            -DBMF_PYENV="${python_versions[$i]}" \
            -mmacos-version-min=10.15 \
            -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
            -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ..
        make -j8

        export RLINK_TYPE=@rpath

        # Patch dependencies
        cd output/bmf/lib
        for file in *.dylib
        do
            # Update install path
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change $PWD/libbmf_module_sdk.dylib @loader_path/libbmf_module_sdk.so "$file"
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change /usr/local/opt/glog/lib/libglog.0.dylib @loader_path/libglog.0.dylib "$file"
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change /usr/local/opt/ffmpeg/lib/libavcodec.58.dylib $RLINK_TYPE/libavcodec.58.dylib "$file"
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change /usr/local/opt/ffmpeg/lib/libavformat.58.dylib $RLINK_TYPE/libavformat.58.dylib "$file"
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change /usr/local/opt/ffmpeg/lib/libavfilter.7.dylib $RLINK_TYPE/libavfilter.7.dylib "$file"
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change /usr/local/opt/ffmpeg/lib/libavdevice.58.dylib $RLINK_TYPE/libavdevice.58.dylib "$file"
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change /usr/local/opt/ffmpeg/lib/libavutil.56.dylib $RLINK_TYPE/libavutil.56.dylib "$file"
            $OSX_XCOMPILE_TOOL_PREFIX-install_name_tool -change /usr/local/opt/ffmpeg/lib/libswresample.3.dylib $RLINK_TYPE/libswresample.3.dylib "$file"

            # Copy all .dylib to .so
            cp "$file" "${file/.dylib/.so}"
        done

        # Copy all third-party dependencies in lib folder
        for file in $OSX_XCOMPILE_ROOT/osx-rootfs/usr/local/lib/*
        do
            if [[ $file == *.dylib ]]
            then
                echo "Copy file: ${file}"
                cp $file .
            fi
        done

        cd ../../..
        cp -r output/bmf/lib/. ../bmf/lib
        cp -r output/bmf/bin ../bmf/bin
        cp -r $OSX_XCOMPILE_ROOT/osx-rootfs/usr/local/include/boost output/bmf/include/
        cp -r $OSX_XCOMPILE_ROOT/osx-rootfs/usr/local/include/glog output/bmf/include/
        cp -r ../3rd_party/json/include/. output/bmf/include/
        cp -r output/bmf/include ../bmf/include

        # Package for specific python version
        cd ..
        /usr/bin/python3.7 setup.py bdist_wheel
        cd dist
        for file in *.whl
        do
            mv "$file" ../output/dist/"${file%-py3-none-any.whl}-cp${python_names[$i]}-none-macosx_10_15_x86_64.whl"
        done
        cd ..

        # Clean up
        rm -rf build_osx
        rm -rf dist
        rm -rf bmf/lib
        rm -rf bmf/bin
        rm -rf bmf/include
        rm -rf byted_bmf.egg-info
    done
fi
