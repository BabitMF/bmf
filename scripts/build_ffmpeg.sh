#!/bin/bash
set -exuo pipefail
DEVICE=cpu
OS=$(uname)
if [ ${OS} == "Linux" ]
then
    . /etc/os-release
elif [ ${OS} == "Darwin" ]
then
    ARCH=$(uname -m)
fi


while [[ $# -gt 0 ]]
do
    arg=$1
    case ${arg} in
        --device=*)
            DEVICE=$(echo ${arg#--device=} | tr 'A-Z' 'a-z')
            ;;
        --arch=*)
            ARCH=${arg#--arch=}
            ;;
        *)
            break
            ;;
    esac
    shift
done


function install_dependencies_linux() {
    if [[ ${NAME} =~ "CentOS" ]]
    then
        yum install -y autoconf automake bzip2 bzip2-devel cmake gcc gcc-c++ git libtool make pkgconfig zlib-devel wget
    elif [[ ${NAME} == "Ubuntu" ]] || [[ ${NAME} =~ "Debian" ]]
    then
        apt install -y autoconf automake bzip2 cmake gcc g++ git libtool make pkg-config wget curl
    fi

}

function install_dependencies_macos() {
    brew install automake git libtool wget
}

function build_nasm_unix() {
    cd $1
    curl -O -L https://download.automotivelinux.org/AGL/mirror/nasm-2.15.05.tar.bz2
    tar xjvf nasm-2.15.05.tar.bz2
    cd nasm-2.15.05
    ./autogen.sh
    ./configure --enable-shared
    make -j $2
    make install
}

function build_yasm_unix() {
    cd $1
    curl -O -L https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
    tar xzvf yasm-1.3.0.tar.gz
    cd yasm-1.3.0
    ./configure
    make -j $2
    make install
}

function build_x264_unix() {
    cd $1
    git clone --branch stable --depth 1 https://code.videolan.org/videolan/x264.git
    cd x264
    ./configure --enable-shared
    make -j $2
    make install
}

function build_x265_unix() {
    cd $1
    git clone --branch stable --depth 7 https://bitbucket.org/multicoreware/x265_git
    cd $1/x265_git/build/linux
    cmake -G "Unix Makefiles" -DENABLE_SHARED:bool=off ../../source
    make -j $2
    make install
}

function build_fdk-aac_unix() {
    cd $1
    git clone --depth 1 https://github.com/mstorsjo/fdk-aac
    cd fdk-aac
    autoreconf -fiv
    ./configure --enable-shared
    make -j $2
    make install
}

function build_mp3lame_unix() {
    cd $1
    curl -O -L https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz
    tar xzvf lame-3.100.tar.gz
    cd lame-3.100
    ./configure --enable-shared --enable-nasm
    make -j $2
    make install
}

function build_opus_unix() {
    cd $1
    curl -O -L https://archive.mozilla.org/pub/opus/opus-1.3.1.tar.gz
    tar xzvf opus-1.3.1.tar.gz
    cd opus-1.3.1
    ./configure --enable-shared
    make -j $2
    make install
}

function build_vpx_unix() {
    cd $1
    git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git
    cd libvpx
    ./configure --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm
    make -j $2
    make install
}

function build_ffmpeg_unix() {
    cd $1
    if [ ! -e ffmpeg-4.4.tar.gz2 ]
    then
        curl -O -L https://ffmpeg.org/releases/ffmpeg-4.4.tar.bz2
    fi
    if [ -d ffmpeg-4.4 ]
    then
        rm -rf ffmpeg-4.4
    fi
    tar xjvf ffmpeg-4.4.tar.bz2
    cd ffmpeg-4.4

    if [ ${OS} == "Linux" ] && [[ ${DEVICE} =~ "gpu" ]] && [ $(uname -m) == "x86_64" ]
    then
        sed -i 's/-gencode arch=compute_30,code=sm_30 -O2/-arch=sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86/g' configure
        sed -i 's/ -ptx//g' configure
    fi

    trap 'cat ffbuild/config.log' ERR
    case $3 in
        x86_64)
            ./configure \
              --pkg-config-flags="--static" \
              --enable-shared \
              --disable-static \
              --extra-libs=-lpthread \
              --extra-libs=-lm \
              --cc='clang -arch x86_64' \
              ${@:4}
            ;;
        arm64)
            ./configure \
              --pkg-config-flags="--static" \
              --enable-shared \
              --disable-static \
              --extra-libs=-lpthread \
              --extra-libs=-lm \
              --cc='clang -arch arm64' \
              ${@:4}
            ;;
        *)
            ./configure \
              --pkg-config-flags="--static" \
              --enable-shared \
              --disable-static \
              --extra-libs=-lpthread \
              --extra-libs=-lm \
              ${@:4}
            ;;
    esac
    trap - ERR

    make -j $2
    make install
}

# cuda12.2 is now supported
function install_cuda_linux() {
    cd $1
    if [[ ${NAME} == "Ubuntu" ]] && [[ ${VERSION_ID} == "20.04" ]]
    then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
	    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
	    dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
	    cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
        apt-get update
        apt-get -y install cuda-toolkit-12-2 cuda-compat-12-2     
	rm -rf cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
	dpkg -r cuda-repo-ubuntu2004-12-2-local
    elif [[ ${NAME} == "Ubuntu" ]] && [[ ${VERSION_ID} == "22.04" ]]
    then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
        mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb            dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
	    cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        apt-get update
        apt-get -y install cuda-toolkit-12-2 cuda-compat-12-2  
    elif [[ ${NAME} =~ "Debian" ]] && [[ ${VERSION_ID} == "11" ]]
    then
        wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-debian11-12-2-local_12.2.0-535.54.03-1_amd64.deb
        dpkg -i cuda-repo-debian11-12-2-local_12.2.0-535.54.03-1_amd64.deb
        cp /var/cuda-repo-debian11-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
        add-apt-repository contrib
        apt-get update
        apt-get -y install cuda-toolkit-12-2 cuda-compat-12-2  
    elif [[ ${NAME} =~ "AlmaLinux" ]] && [[ ${VERSION_ID} == "8.10" ]]
    then
        original_ld_lib_path=$LD_LIBRARY_PATH
        unset LD_LIBRARY_PATH
        dnf install epel-release wget -y
        dnf install dkms -y
        wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-rhel8-12-2-local-12.2.0_535.54.03-1.x86_64.rpm
        rpm -i cuda-repo-rhel8-12-2-local-12.2.0_535.54.03-1.x86_64.rpm
        dnf clean all
        dnf -y module install nvidia-driver:latest-dkms
        dnf -y install cuda
        export LD_LIBRARY_PATH=$original_ld_lib_path
    fi
    export PATH=${PATH}:/usr/local/cuda/bin
    cd -
}

#XXX: only ubuntu
function install_cvcuda_linux() {
    cd $1
    mkdir -p cvcuda_source
    cd cvcuda_source
    wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-lib-0.3.1_beta-cuda12-x86_64-linux.deb
    wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-dev-0.3.1_beta-cuda12-x86_64-linux.deb
    wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-python3.8-0.3.1_beta-cuda12-x86_64-linux.deb
    wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv_python-0.3.1_beta-cp38-cp38-linux_x86_64.whl
    apt install ./nvcv-lib-0.3.1_beta-cuda12-x86_64-linux.deb ./nvcv-dev-0.3.1_beta-cuda12-x86_64-linux.deb ./nvcv-python3.8-0.3.1_beta-cuda12-x86_64-linux.deb
    pip3 install ./nvcv_python-0.3.1_beta-cp38-cp38-linux_x86_64.whl
    export PYTHONPATH=/usr/local/lib/python3.8/dist-packages/nvcv_python
    cd -
    rm -rf cvcuda_source

    cd $1
    cd ..
}

function install_trt_linux() {
    cd $1
    mkdir -p trt
    cd trt
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
    version="8.6.1.6"
    arch=$(uname -m)
    cuda="cuda-12.0"
    tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz
    python3 -m pip install --upgrade pip
    cd TensorRT-${version}/python
    python3 -m pip install tensorrt-*-cp38-none-linux_x86_64.whl --force-reinstall
    python3 -m pip install tensorrt_lean-*-cp38-none-linux_x86_64.whl --force-reinstall
    python3 -m pip install tensorrt_dispatch-*-cp38-none-linux_x86_64.whl --force-reinstall
    cd -
#    cd TensorRT-${version}/uff
#    python3 -m pip install uff-0.6.9-py2.py3-none-any.whl --force-reinstall
#    cd -
    cd TensorRT-${version}/graphsurgeon

    python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl -force-reinstall
    cd -
    cd TensorRT-${version}/onnx_graphsurgeon
        
    python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl -force-reinstall
    cd ..
    rm -rf python uff graphsurgeon onnx_graphsurgeon
    cd ..
    mv TensorRT-${version} /usr/local/
    cd ..
    rm -rf trt
    export LD_LIBRARY_PATH=/usr/local/TensorRT-${version}/lib
    export PATH=$PATH:/usr/local/TensorRT-${version}/bin

    cd $1
    cd ..
}

function build_ffnvcodec_linux() {
    cd $1
    git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
    cd nv-codec-headers
    git checkout n10.0.26.2
    make install
}

function install_cudnn_linux() {
    rm /etc/apt/sources.list.d/cuda*
    cd $1
    mkdir cudnn
    cd cudnn
    if [[ ${NAME} == "Ubuntu" ]] && [[ ${VERSION_ID} == "20.04" ]]
    then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update
        apt-get -y install libcudnn8 libcudnn8-dev
    elif [[ ${NAME} == "Ubuntu" ]] && [[ ${VERSION_ID} == "22.04" ]]
    then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update
        apt-get -y install libcudnn8 libcudnn8-dev
    elif [[ ${NAME} =~ "Debian" ]] && [[ ${VERSION_ID} == "11" ]]
    then
        wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update
        apt-get -y install libcudnn8 libcudnn8-dev
    fi
    cd -
    rm -rf cudnn
}


function check_lib() {
    cd $1
    if [ ! -e ffmpeg-4.4 ]
    then
        curl -O -L https://ffmpeg.org/releases/ffmpeg-4.4.tar.bz2
        tar xjvf ffmpeg-4.4.tar.bz2
    fi

    cd ffmpeg-4.4
    str="\-\-enable-lib"$2
    ./configure --help | grep ''${str}''
    if [ $? -eq 0 ]
    then
        return 0
    fi
    return 1
}

function change_ffmpeg_deps_macos() {
    otool -L $1 | awk '{if(NF>1){print($1);}}' | grep ffmpeg_x86_64 | while read old_dep
    do
        new_dep=$(echo ${old_dep} | sed 's/ffmpeg_x86_64/ffmpeg_'"$3"'/g')
        install_name_tool -change ${old_dep} ${new_dep} $1
    done

    if [ $2 == "lib" ]
    then
        otool -D $1 | grep -v ":$" | grep ffmpeg_x86_64 | while read old_id
        do
            new_id=$(echo ${old_id} | sed 's/ffmpeg_x86_64/ffmpeg_'"$3"'/g')
            install_name_tool -id ${new_id} $1
        done
    fi
}

if [ ${OS} == "Linux" ] || [ ${OS} == "Darwin" ]
then
    disable_asm="--disable-x86asm"
    ffmpeg_opts="--enable-gpl --enable-nonfree"
    mkdir -p ffmpeg_source
    trap "cd $(pwd) && rm -rf ffmpeg_source" EXIT

    if [ ${OS} == "Linux" ]
    then
        (install_dependencies_linux)
        if [[ ${DEVICE} =~ "gpu" ]] && [ $(uname -m) == "x86_64" ]
        then
            install_cuda_linux $(pwd)/ffmpeg_source
            if [ ${DEVICE} == "gpu_all" ]
            then
                (install_cudnn_linux $(pwd)/ffmpeg_source)
                install_trt_linux $(pwd)/ffmpeg_source
                install_cvcuda_linux $(pwd)/ffmpeg_source
            fi

            (build_ffnvcodec_linux $(pwd)/ffmpeg_source)
            ffmpeg_opts="${ffmpeg_opts} --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64"
        fi
        cores=$(nproc)
    else
        (install_dependencies_macos)
        cores=$(sysctl -n hw.logicalcpu)
    fi

    for arg in $@
    do
        (build_${arg}_unix $(pwd)/ffmpeg_source ${cores})
        if [ ${arg} == "nasm" ] || [ ${arg} == "yasm" ]
        then
            disable_asm=""
        fi

        set +e
        (check_lib $(pwd)/ffmpeg_source ${arg})
        if [ $? -eq 0 ]
        then
            ffmpeg_opts="${ffmpeg_opts} --enable-lib${arg}"
        fi
        set -e
    done

    ffmpeg_opts="${ffmpeg_opts} ${disable_asm}"


    if [ ${OS} == "Linux" ]
    then
        printf "ffmpeg_opts: %s\n" ${ffmpeg_opts}
        (build_ffmpeg_unix $(pwd)/ffmpeg_source ${cores} "" ${ffmpeg_opts})
    else
        if [[ ${ARCH} = universal* ]]
        then
            for arch in "x86_64" "arm64"
            do
                ffmpeg_opts_mac="${ffmpeg_opts} --prefix=$(pwd)/ffmpeg_${arch} --enable-cross-compile --arch=${arch}"
                (build_ffmpeg_unix $(pwd)/ffmpeg_source ${cores} ${arch} ${ffmpeg_opts_mac})
            done

            mkdir -p ffmpeg_${ARCH}/{bin,lib}
            cp -r ffmpeg_x86_64/{include,share} ffmpeg_${ARCH}
            cp -r ffmpeg_x86_64/lib/pkgconfig ffmpeg_${ARCH}/lib/pkgconfig
            for bin in `find ffmpeg_x86_64/bin -maxdepth 1 -not -type d`
            do
                lipo -create ffmpeg_x86_64/bin/$(basename ${bin}) ffmpeg_arm64/bin/$(basename ${bin}) -output ffmpeg_${ARCH}/bin/$(basename ${bin})
            done
            for lib in `find ffmpeg_x86_64/lib -maxdepth 1 -not -type d`
            do
                lipo -create ffmpeg_x86_64/lib/$(basename ${lib}) ffmpeg_arm64/lib/$(basename ${lib}) -output ffmpeg_${ARCH}/lib/$(basename ${lib})
            done

            for bin in `find ffmpeg_${ARCH}/bin -maxdepth 1 -not -type d`
            do
                change_ffmpeg_deps_macos ${bin} "bin" ${ARCH}
            done
            for lib in `find ffmpeg_${ARCH}/lib -maxdepth 1 -not -type d`
            do
                change_ffmpeg_deps_macos ${lib} "lib" ${ARCH}
            done
            for pc_file in `find ffmpeg_${ARCH}/lib/pkgconfig -maxdepth 1 -not -type d`
            do
                sed -i '' 's/ffmpeg_x86_64/ffmpeg_'"${ARCH}"'/g' ${pc_file}
            done
        else
            ffmpeg_opts="${ffmpeg_opts} --prefix=$(pwd)/ffmpeg_${ARCH} --enable-cross-compile --arch=${ARCH}"
            (build_ffmpeg_unix $(pwd)/ffmpeg_source ${cores} ${ARCH} ${ffmpeg_opts})
        fi
    fi
else
    printf "the system %s is not supported!" ${OS}
    exit 1
fi
