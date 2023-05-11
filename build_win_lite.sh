#!/bin/bash

set -eux

#git submodule update --init --recursive

HOST=$(uname | tr 'A-Z' 'a-z')
BUILD_TYPE="Release"
BUILD_DIR=build_win_lite
OUTPUT_DIR=output

export SCRIPT_EXEC_MODE=win
export WIN_XCOMPILE_ARCH=x86_64
export WIN_XCOMPILE_ROOT=$(pwd)/3rd_party/win_rootfs
export PLATFORM_NAME=x64
export USE_BMF_FFMPEG=0

[ $# -gt 0 ] && {
    for arg in $*
    do
        case $arg in
            debug)
                BUILD_TYPE="Debug"
                ;;
            clean)
                rm -rf ${BUILD_DIR}
                exit
                ;;
	    --platform=*)
		export PLATFORM_NAME=${arg#--platform=}
		;;
            bmf_ffmpeg)
                export USE_BMF_FFMPEG=1
		;;
            *)
                printf "arg:%s is not supported.\n" ${arg}
                exit 1
                ;;
        esac
    done
}

if [ "$USE_BMF_FFMPEG" = "1" ] && [ ! -d "3rd_party/ffmpeg_bin/win/build" ]
then
    git config --global http.sslVerify false
    git submodule init "3rd_party/ffmpeg_bin"
    git submodule update
    
    echo "Extracting BMF's FFMPEG"
    cd 3rd_party/ffmpeg_bin/win/
    tar xvf ffmpeg5.0.tar.gz
    cd ../../..
else
	echo "Skip extracting ffmpeg"
fi

source ./version.sh

[ -d ${OUTPUT_DIR} ] && rm -rf ${OUTPUT_DIR}/* || mkdir -p ${OUTPUT_DIR}
[ -d ${BUILD_DIR} ] && rm -rf ${BUILD_DIR}/* || mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

if [[ ${HOST} =~ mingw ]]
then
    for preset in x86-Debug x64-Debug x86-Release x64-Release
    do
	    if [ "$USE_BMF_FFMPEG" = "1" ] && [ ! -d "3rd_party/ffmpeg_bin/win/build" ]
	    then
		    if [ $(echo ${preset} | awk -F'-' '{print $1}') == "x86" ]
		    then
			    dir=x86
		    else
			    dir=x86_64
		    fi

		    cp -r ../3rd_party/ffmpeg_bin/win/build/${dir}/lib/. /usr/local/lib/
		    cp -r ../3rd_party/ffmpeg_bin/win/build/${dir}/include/. /usr/local/include/
		    cp -r ../3rd_party/ffmpeg_bin/win/build/${dir}/bin/. /usr/local/bin/
	    fi

        #need to add '-DRUN_HAVE_STD_REGEX=0 -DRUN_HAVE_POSIX_REGEX=0' to cmake, see https://github.com/google/benchmark/issues/773
        [ -d ${preset} ] && rm -rf ${preset}/* || mkdir -p ${preset}

	(
	    cd ${preset}
	    cmake -G 'Visual Studio 16 2019' --preset ${preset} \
            -DCMAKE_VERBOSE_MAKEFILE=ON \
            -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE \
            -DBUILD_SHARED_LIBS=TRUE \
            -DCMAKE_TOOLCHAIN_FILE=../../cmake/win-toolchain.cmake \
            -DBMF_ENABLE_PYTHON=OFF \
            -DBMF_ENABLE_GLOG=OFF \
            -DBMF_ENABLE_MOBILE=OFF \
            -DBMF_ENABLE_FFMPEG=OFF \
            -DBMF_ENABLE_CUDA=OFF \
            -DRUN_HAVE_STD_REGEX=0 \
            -DRUN_HAVE_POSIX_REGEX=0 \
            -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
            -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ../..

	    cmake --build . --config $(echo ${preset} | awk -F'-' '{print $2}')

	    cp -r output ../../${OUTPUT_DIR}/${preset}
        )
    done

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
        -DBMF_ENABLE_PYTHON=OFF \
        -DBMF_ENABLE_GLOG=OFF \
        -DBMF_ENABLE_MOBILE=OFF \
        -DBMF_ENABLE_FFMPEG=OFF \
        -DBMF_ENABLE_CUDA=OFF \
        -DRUN_HAVE_STD_REGEX=0 \
        -DRUN_HAVE_POSIX_REGEX=0 \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DBMF_BUILD_VERSION=${BMF_BUILD_VERSION} \
        -DBMF_BUILD_COMMIT=${BMF_BUILD_COMMIT} ..

    make -j$(nproc)

    cp -r ${BUILD_DIR}/output ${OUTPUT_DIR}
fi
