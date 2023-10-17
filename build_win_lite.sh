#!/bin/bash
set -x
set -eux

#git submodule update --init --recursive

MSVC_VERSION=""
HOST=$(uname | tr 'A-Z' 'a-z')
BUILD_TYPE="Release"
BUILD_DIR=build_win_lite
OUTPUT_DIR=output
COMPILE_ARCH=""
preset=""
export SCRIPT_EXEC_MODE=win

export WIN_XCOMPILE_ROOT=$(pwd)/3rd_party/win_rootfs
export PLATFORM_NAME=x64
export USE_BMF_FFMPEG=0


[ $# -gt 0 ] && {
    for arg in "$@"; do
    case $arg in
        clean)
        rm -rf ${BUILD_DIR}
        exit
        ;;
        --msvc=2013|--msvc=2015|--msvc=2017|--msvc=2019|--msvc=2022)
        MSVC_VERSION=${arg#--msvc=}
        ;;
        --preset=x86-Debug|--preset=x86-Release|--preset=x64-Debug|--preset=x64-Release)
        preset=${arg#--preset=}
        ;;
        bmf_ffmpeg)
        USE_BMF_FFMPEG="ON"
        ;;
        *)
        printf "arg:%s is not supported.\n" "${arg}"
        exit 1
        ;;
    esac
    done
}

if [ -z "$MSVC_VERSION" ]; then
    printf "Please specify the MSVC version using --msvc=[2013,2015,2017,2019,2022].\n"
    exit 1
fi

if [ -z "$preset" ]; then
    printf "Please specify the MSVC arch preset using --preset=[x86-Debug,x86-Release,x64-Debug,x64-Release].\n"
    exit 1
fi

if [ "$preset" = "x64-Debug" ] || [ "$preset" = "x86-Debug" ]; then
    BUILD_TYPE="Debug"
fi

if [ "$preset" = "x86-Debug" ] || [ "$preset" = "x86-Release" ]; then
    export WIN_XCOMPILE_ARCH=x86
fi

if [ "$preset" = "x64-Debug" ] || [ "$preset" = "x64-Release" ]; then
    export WIN_XCOMPILE_ARCH=x64
fi

case $MSVC_VERSION in
    2013)
        CMAKE_GENERATOR="Visual Studio 12 2013"
        ;;
    2015)
        CMAKE_GENERATOR="Visual Studio 14 2015"
        ;;
    2017)
        CMAKE_GENERATOR="Visual Studio 15 2017"
        ;;
    2019)
        CMAKE_GENERATOR="Visual Studio 16 2019"
        ;;
    2022)
        CMAKE_GENERATOR="Visual Studio 17 2022"
        ;;
    *)
        printf "Unsupported MSVC version: %s\n" $MSVC_VERSION
        exit 1
        ;;
esac
git submodule update --init --recursive

if [ ! -d "3rd_party/win_rootfs" ]
then
    (cd 3rd_party/ && wget https://github.com/BabitMF/bmf/releases/download/files/win_rootfs.tar.gz && tar zvxf win_rootfs.tar.gz)
fi
source ./version.sh

[ -d ${OUTPUT_DIR} ] && rm -rf ${OUTPUT_DIR}/* || mkdir -p ${OUTPUT_DIR}
[ -d ${BUILD_DIR} ] && rm -rf ${BUILD_DIR}/* || mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

#x86-Debug x64-Debug x86-Release x64-Release
if [[ ${HOST} =~ msys_nt || ${HOST} =~ mingw ]]; then
    echo "Building ${preset} ${BUILD_TYPE}"

    [ -d ${preset} ] && rm -rf ${preset}/* || mkdir -p ${preset}

(
    cd ${preset}
    cmake -DCMAKE_VERBOSE_MAKEFILE=ON -G "${CMAKE_GENERATOR}" --preset ${preset} \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE \
        -DBUILD_SHARED_LIBS=TRUE \
        -DCMAKE_TOOLCHAIN_FILE=../../cmake/win-toolchain.cmake \
        -DBMF_ENABLE_PYTHON=ON \
        -DBMF_ENABLE_MOBILE=OFF \
        -DBMF_ENABLE_FFMPEG=${USE_BMF_FFMPEG} \
        -DBMF_ENABLE_CUDA=OFF \
        -DRUN_HAVE_STD_REGEX=0 \
        -DRUN_HAVE_POSIX_REGEX=0 \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
        -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ../..

    )

    cat >../output/current_revision <<EOF
revision:$(git rev-parse HEAD)
version:none
pub data:$(date -u +"%Y-%m-%d %H:%M:%S")
arch:x86_64
region:none
source code repo name:$(git remote get-url origin)
compiled by msvc, build command: $0 $@
EOF
else
    cmake -A ${PLATFORM_NAME} \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE \
        -DBUILD_SHARED_LIBS=TRUE \
        -DCMAKE_TOOLCHAIN_FILE=../cmake/win-toolchain.cmake \
        -DBMF_ENABLE_PYTHON=ON \
        -DBMF_ENABLE_MOBILE=OFF \
        -DBMF_ENABLE_FFMPEG=${USE_BMF_FFMPEG} \
        -DBMF_ENABLE_CUDA=OFF \
        -DRUN_HAVE_STD_REGEX=0 \
        -DRUN_HAVE_POSIX_REGEX=0 \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
        -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ..

    make -j$(nproc)

    cp -r ${BUILD_DIR}/output ${OUTPUT_DIR}
fi
