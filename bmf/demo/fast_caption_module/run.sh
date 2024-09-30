#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../../../build/output:../../../build/output/bmf/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../3rd_party/tensorrt_llm/lib:../../../3rd_party/libtorch/lib:../../../3rd_party/opencv:/root/bmf_v3/3rd_party/lib

python3 test/test.py -g=0 -l=libbcm -v=example/videos
