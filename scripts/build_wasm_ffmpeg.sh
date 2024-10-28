#!/bin/bash
# set -exuo pipefail
OS=$(uname)
INSTALL_DIR=$(pwd)/ffmpeg
if [ ${OS} != "Linux" ]
then
    echo "Not support other platform yet".
    exit
fi

function build_x264() {
    cd $1
    git clone https://code.videolan.org/videolan/x264.git
    cd x264
    CONF_FLAGS=(
        --prefix=$INSTALL_DIR           # lib installation dir
        --host=x86-gnu                  # use x86 linux host
        --enable-static                 # build static library
        --disable-cli                   # disable cli build
        --disable-asm                   # disable assembly
        --disable-thread
        --extra-cflags="-fPIC -pthread"
    )
    emconfigure ./configure "${CONF_FLAGS[@]}"
    emmake make install-lib-static -j
}

function build_x265() {
    cd $1
    # x265 needs some edits. 
    # Should we use a personal repo?
    git clone --branch stable --depth 7 https://github.com/ruiqurm/x265
    cd source
    mkdir build
    cd build
    CONF_FLAGS=(
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
        -DENABLE_LIBNUMA=OFF
        -DENABLE_SHARED=OFF
        -DENABLE_CLI=OFF
        -DCMAKE_CXX_FLAGS="-msimd128"
        -DCMAKE_C_FLAGS="-msimd128"
    )
    emmake cmake .. ${CONF_FLAGS[@]}
    mmake make -j
    emmake make install -j
}

function build_ffmpeg() {
    cd $1
    curl -O -L https://ffmpeg.org/releases/ffmpeg-4.4.tar.bz2
    tar xjvf ffmpeg-4.4.tar.bz2
    cd ffmpeg-4.4
    export CFLAGS="-I$INSTALL_DIR/include $CFLAGS"
    export CXXFLAGS="$CFLAGS"
    export LDFLAGS="-L$INSTALL_DIR/lib $LDFLAGS $CFLAGS"
    export EM_PKG_CONFIG_PATH=$EM_PKG_CONFIG_PATH:$INSTALL_DIR/lib/pkgconfig:$EMSDK/upstream/emscripten/system/lib/pkgconfig
    export EM_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake
    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$EM_PKG_CONFIG_PATH
    CONF_FLAGS=(
        --prefix=$INSTALL_DIR         # lib installation dir
        --target-os=none              # disable target specific configs
        --arch=x86_32                 # use x86_32 arch
        --disable-pthreads
        --disable-os2threads
        --disable-w32threads
        --enable-libx264
        --enable-gpl
        --enable-cross-compile        # use cross compile configs
        --disable-asm                 # disable asm
        --disable-stripping           # disable stripping as it won't work
        --disable-programs            # disable ffmpeg, ffprobe and ffplay build
        --disable-doc                 # disable doc build
        --disable-debug               # disable debug mode
        --disable-runtime-cpudetect   # disable cpu detection
        --disable-autodetect          # disable env auto detect
        --nm=emnm
        --ar=emar
        --ranlib=emranlib
        --cc=emcc
        --cxx=em++
        --objcc=emcc
        --dep-cc=emcc
        --extra-cflags="-fPIC -pthread"
    )
    emconfigure ./configure "${CONF_FLAGS[@]}"
    emmake make -j
    emmake make install
}



BUILD_PATH=$(pwd)/ffmpeg/build
mkdir -p $BUILD_PATH
build_x264 $BUILD_PATH
build_ffmpeg $BUILD_PATH
