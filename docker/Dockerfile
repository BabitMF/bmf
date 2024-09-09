FROM ubuntu:20.04
LABEL org.opencontainers.image.source=https://github.com/babitmf/bmf
ARG DEVICE=gpu_all
ARG GITHUB_ACTIONS

ENV GITHUB_ACTIONS=${GITHUB_ACTIONS}
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all
# # required
RUN    apt update \
    && apt install -y ca-certificates make git pkg-config libssl-dev cmake python3 python3-dev python3-pip binutils-dev libgoogle-glog-dev wget gcc g++ yasm nasm apt-utils \
    && apt autoremove cmake -y \
    && wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-Linux-x86_64.tar.gz && tar -zvxf cmake-3.26.4-Linux-x86_64.tar.gz \
    && rm -rf cmake-3.26.4-linux-x86_64/man && cp -r cmake-3.26.4-linux-x86_64/* /usr/local && rm -rf cmake-*

# optional
# RUN    apt install -y libopencv-dev libomp-dev libtorch3-dev

RUN pip3 install setuptools timeout_decorator opencv-python onnx onnxruntime-gpu numpy cuda-python
# torch
RUN pip3 install torch torchvision torchaudio
RUN rm -rf /var/cache/apk/* && \
    rm -rf /root/.cache/pip/*
COPY ./ /root/bmf/
WORKDIR /root/bmf/
# ffmpeg installed via apt has no development libraries, so we build it from source

RUN ./scripts/build_ffmpeg.sh --device=${DEVICE} x264 x265 fdk-aac opus nasm \
    && rm -rf ffmpeg_source

ARG CMAKE_ARGS="-DBMF_ENABLE_CUDA=OFF -DBMF_ENABLE_BREAKPAD=ON"
RUN ./build.sh && rm -rf .git && ./build.sh clean

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/bmf/output/bmf/lib:/usr/local/TensorRT-8.6.1.6/lib:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat
ENV PATH=${PATH}:/root/bmf/output/bmf/bin:/usr/local/cuda/bin:/usr/local/TensorRT-8.6.1.6/bin
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/dist-packages/nvcv_python:/root/bmf/output
