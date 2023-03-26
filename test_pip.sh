#!/bin/bash

# This script is to prepare for testing the correctness of pypi installation

# Copy related dependencies for patching
rm -rf dbglib
mkdir dbglib
cp -r ./3rd_party/ffmpeg_bin/linux/build/lib/. dbglib
cp /usr/lib/x86_64-linux-gnu/libboost_python37.so.1.67.0 dbglib
cp /usr/lib/x86_64-linux-gnu/libboost_numpy37.so.1.67.0 dbglib
cp /usr/lib/x86_64-linux-gnu/libboost_system.so.1.67.0 dbglib
cp /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.67.0 dbglib
cp /usr/lib/x86_64-linux-gnu/libgflags.so.2.2 dbglib
cp /usr/lib/x86_64-linux-gnu/libglog.so.0 dbglib
cp /usr/lib/x86_64-linux-gnu/libunwind.so.8 dbglib

# Patch rpath for dependencies
cd dbglib
for libfile in *.so*
do
    if [ -f "$libfile" ]; then 
        patchelf --set-rpath '$ORIGIN' "$libfile"
        echo "Patch $libfile"
    fi
done
cd ..

# PYPI package
cp -r dbglib/. output/bmf/lib
cp -r output/bmf/lib bmf
cp -r output/bmf/bin bmf
cp -r output/bmf/include bmf
cp bmf/engine/c_engine/BUILTIN_CONFIG.json bmf
cp output/bmf/lib/libcopy_module.so bmf/example/c_module
/usr/bin/python3.7 setup.py bdist_wheel

# Install package for testing
cd dist
for whlfile in *.whl
do
    /usr/bin/python3.7 -m pip install "$whlfile"
done
cd ..
