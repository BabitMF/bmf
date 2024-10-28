#! /bin/bash

git checkout n4.4.5
# INSTALL_DIR=/root/bmf/output
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
emconfigure ./configure --enable-gpl --enable-libx264 --enable-libx265 "${CONF_FLAGS[@]}" 
emmake make -j
emmake make install