#! /bin/bash
# INSTALL_DIR=/root/bmf/output
CONF_FLAGS=(
  --prefix=$INSTALL_DIR           # lib installation dir
  --host=x86-gnu                  # use x86 linux host
  --enable-static                 # build static library
  --disable-cli                   # disable cli build
  --disable-asm                   # disable assembly
  --disable-thread
  --enable-pic
)
emconfigure ./configure "${CONF_FLAGS[@]}"
emmake make install-lib-static -j