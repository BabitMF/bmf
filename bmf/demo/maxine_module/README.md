# Introduction to Maxine Module

NVIDIA Maxine is a suite of GPU-accelerated AI SDKs and cloud-native microservices for deploying AI features that enhance audio, video, and augmented reality effects in real time.

This module only contains video effects currently. The audio and augmented reality effects will be brought in the future.

## Video Effects

Video Effects have below features:

- Virtual Background
- Super Resolution (up to 4X scaling factor)
- Upscaler (up to 4X scaling factor)
- Artifact Reduction
- Video Noise Removal

The `test_maxine_module.py` illustrates how to use all effects. You can also find more details from the online [documentation](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html)

## Build the module

### Prerequisites

- OpenCV

```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip && rm opencv.zip && unzip opencv_contrib.zip && rm opencv_contrib.zip
cd opencv-4.x
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.x/modules .. && cmake --build . --config Release -- -j
make install
```

- CUDA 11.8
- TensorRT 8.5.1.7

### Build

You should download the Video Effect SDK from the [NVIDIA NGC website](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/maxine/resources/maxine_linux_video_effects_sdk_ga) first. 

```
export LD_LIBRARY_PATH=/path/to/maxine/folder/lib:$LD_LIBRARY_PATH
mkdir build
cd build
cmake .. -DMAXINE_DIR=/path/to/maxine/folder
make
```

## Test the effects

The `test_maxine_module.py` file lists all effects this module supports currently.

You should provide the `model_dir` for the first parameter. The value should be `/path/to/maxine/folder/lib/models`.


```
python test_maxine_module.py model_dir
```

For each effect, the effect's option provides some default values that is enough to show the effects. For more options of each effect, you can find them in the `maxine_module.cpp` or the online [documentation](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-programming-guide/index.html).