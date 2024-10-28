#! /bin/bash

cd source
mkdir build
cd build

# INSTALL_DIR=/root/bmf/output
CONF_FLAGS=(
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
  -DENABLE_LIBNUMA=OFF
  -DENABLE_CLI=OFF
  -DENABLE_PIC=ON
)
CXX_FLAGS="-msimd128 -fPIC"
C_FLAGS="-msimd128 -fPIC"

emmake cmake .. "${CONF_FLAGS[@]}" -DCMAKE_CXX_FLAGS="$CXX_FLAGS" -DCMAKE_C_FLAGS="$C_FLAGS"
emmake make -j
emmake make install -j