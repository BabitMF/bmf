# About This Demo

The GPU transcoding and filter module demo shows:
1. Common video/image filters in BMF accelerated by GPU
2. How to write GPU modules in BMF

The demo builds a transcoding pipeline which fully runs on GPU:

decode->scale->flip->rotate->crop->blur->encode

The steps described in this document is to help you to run the demo on your own machine. There is also a ipynb provided in this directory (`gpu_transcoding_filter_module.ipynb`), if you want to have a quick try, you can run it on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabitMF/bmf/blob/master/bmf/demo/gpu_module/gpu_module_demo_colab.ipynb).

## Dependencies
The demo has the following dependencies:
* Python >= 3.8
* CUDA >= 11.0
* ffmpeg >= 4.2
* CV-CUDA >= 0.2.1

# Get Started

## 1. Environment Setup
*   install BMF
*   install dependencies: ffmpeg, cv-cuda
*   make sure the GPU environment is ready

### 1-1 pip install BMF packages

Before installing BMF, please make sure that you have installed Python and pip. It is recommended to use Python 3.8 or newer versions for better compatibility.

To install a GPU supported version of BMF:

```Bash
pip3 install -i https://test.pypi.org/simple/ BabitMF-GPU
%env LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/bmf/lib
```

### 1-2 install the FFmpeg libraries

Part of feature in BMF framework utilizes the FFmpeg demuxer/muxer/codec and filter as the built-in modules for video processing, especially, BMF utilizes GPU's hardware codec through ffmpeg. **The capability of ffmpeg is needed in this demo,it's neccessary for users to install supported FFmpeg libraries before using BMF.**

On Ubuntu, we can install ffmpeg through apt.
```Bash
sudo apt update
sudo apt install ffmpeg libdw1
```
If you are using Rocky Linux, please refer to [this article](https://citizix.com/how-to-install-ffmpeg-on-rocky-linux-alma-linux-8/) on how to install ffmpeg.

Make sure the ffmpeg version is >= 4.2

```Bash
ffmpeg -version
```

### 1-3 install CV-CUDA

The GPU modules are implemented using CV-CUDA, **it's necessary to install CV-CUDA if you want to run the modules.** Note that we need CV-CUDA version >= 0.2.1 to make the modules work.

To install CV-CUDA, we need to collect the pre-built binaries from github then install them using apt.
```Bash
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-lib-0.3.1_beta-cuda12-x86_64-linux.deb
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-dev-0.3.1_beta-cuda12-x86_64-linux.deb
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv-python3.10-0.3.1_beta-cuda12-x86_64-linux.deb
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.3.1-beta/nvcv_python-0.3.1_beta-cp310-cp310-linux_x86_64.whl

sudo apt install ./nvcv-lib-0.3.1_beta-cuda12-x86_64-linux.deb ./nvcv-dev-0.3.1_beta-cuda12-x86_64-linux.deb ./nvcv-python3.10-0.3.1_beta-cuda12-x86_64-linux.deb
```

Verify that CV-CUDA has been installed properly.

```Python
import cvcuda
print(cvcuda.__version__)
```

## 2. BMF GPU Transcoding & Filter Demo
 Now let's set up a gpu transcoding pipeline with common filters. The pipeline will be complete run on GPU, which means the data does not need to be copied back to CPU. We should always avoid CPU-GPU data movement as much as possible, this is an important practice in terms of performance.

 ### 2-1 Run the demo

The example GPU modules are implemented in the following files:

*   scale_gpu.py
*   flip_gpu.py
*   rotate_gpu.py
*   crop_gpu.py
*   blur_gpu.py

You can tell what the module does by its name. Please refer to `bmf/bmf/docs/example/Example_GpuModule.md` for detailed documentation of the GPU modules. There is a Python script `test_gpu_module.py` in the module directory showing how to run these modules

```Bash
python3 test_gpu_module.py
```

The output video can by played using ffplay.
```Bash
ffplay -i ./output.mp4
```