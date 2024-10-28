FROM emscripten/emsdk

ENV DEBIAN_FRONTEND=noninteractive
ENV INSTALL_DIR=/opt

ENV CFLAGS="-I$INSTALL_DIR/include $CFLAGS"
ENV CXXFLAGS="$CFLAGS"
ENV LDFLAGS="-L$INSTALL_DIR/lib $LDFLAGS $CFLAGS"
ENV EM_PKG_CONFIG_PATH=$EM_PKG_CONFIG_PATH:$INSTALL_DIR/lib/pkgconfig:$EMSDK/upstream/emscripten/system/lib/pkgconfig
ENV EM_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake
ENV PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$EM_PKG_CONFIG_PATH

# build basic environment
RUN sed -i 's/archive.ubuntu.com/mirrors.bfsu.edu.cn/g' /etc/apt/sources.list
RUN apt update \
&& apt install -y make git pkg-config libssl-dev cmake python3 python3-pip
RUN git config --global http.proxy http://172.27.224.1:10808 && git config --global https.proxy http://172.27.224.1:10808

# download files
RUN cd /root/ && git clone https://code.videolan.org/videolan/x264 && git clone https://github.com/ruiqurm/x265.git && \
    git clone https://github.com/FFmpeg/FFmpeg

# copy files
COPY ./ /root/bmf/

# build ffmpeg, x264 and x265 as shared library
RUN cd /root/x264 && \
    cp /root/bmf/docker/wasm_edv/build_x264.sh . && ./build_x264.sh
RUN cd /root/x265 && \
    cp /root/bmf/docker/wasm_edv/build_x265.sh . && ./build_x265.sh

# RUN cd /root/bmf  && cp -r output/lib/* /emsdk/upstream/emscripten/cache/sysroot/lib/wasm32-emscripten/ &&\
#     cp -r output/include/* /emsdk/upstream/emscripten/cache/sysroot/include/
RUN cd /root/FFmpeg && \ 
    cp /root/bmf/docker/wasm_edv/build_ffmpeg.sh . && ./build_ffmpeg.sh 

# copy BUILTIN_CONFIG.json and download big_bunny_10s_30fps.mp4 
# for latter compiling
# (only use these in developing)

RUN cp /root/bmf/bmf/c_modules/meta/BUILTIN_CONFIG.json .

# call ./build_wasm.sh to compile
# And call python3 -m http.server 
# open xxx.html to run wasm application locally.
ENV FFMPEG_ROOT_PATH=/opt
ENV FFMPEG_ROOT_PATH=/opt
ENV LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH
CMD ["/bin/bash"]


