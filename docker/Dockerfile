FROM ubuntu:20.04
LABEL org.opencontainers.image.source=https://github.com/babitmf/bmf
ARG DEVICE=gpu
ARG GITHUB_ACTIONS

ENV GITHUB_ACTIONS=${GITHUB_ACTIONS}
ENV DEBIAN_FRONTEND=noninteractive
# required
RUN    apt update \
    && apt install -y ca-certificates make git pkg-config libssl-dev cmake python3 python3-dev python3-pip binutils-dev libgoogle-glog-dev wget gcc g++ yasm nasm apt-utils
# optional
# RUN    apt install -y libopencv-dev libomp-dev libtorch3-dev

RUN pip3 install setuptools timeout_decorator opencv-python onnx onnxruntime-gpu numpy cuda-python
# torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /var/cache/apk/* && \
    rm -rf /root/.cache/pip/*
COPY ./ /root/bmf/
WORKDIR /root/bmf/
# ffmpeg installed via apt has no development libraries, so we build it from source

RUN ./scripts/build_ffmpeg.sh --device=$DEVICE x264 x265 fdk-aac opus \
    && rm -rf ffmpeg_source

ARG CMAKE_ARGS="-DBMF_ENABLE_CUDA=OFF -DBMF_ENABLE_BREAKPAD=ON"
RUN ./build.sh && rm -rf .git && ./build.sh clean

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/bmf/output/bmf/lib:/usr/local/cuda/lib64:/usr/local/TensorRT-8.6.1.6/lib:/usr/local/lib
ENV PATH=${PATH}:/root/bmf/output/bmf/bin:/usr/local/cuda/bin:/usr/local/TensorRT-8.6.1.6/bin
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/dist-packages/nvcv_python:/root/bmf/output
