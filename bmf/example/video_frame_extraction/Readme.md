# About this demo

The video frame extraction acceleration demo shows:
1. BMF flexible capability of:

   *   Multi-language programming，we can see multi-language module work together in the demo
   *   Ability extend easily, there are new C++, Python modules added simply
   *   FFmpeg ability fully compatible

2. Hardware acceleration quickly enablement and CPU/GPU pipeline support

   *   Heterogeneous pipeline is supported in BMF, such as process between CPU and GPU
   *   Usefull hardware color space convertion in BMF

   

The graph or pipeline looks like:

* Normal:

  Video Decode  ==> FPS filter frame extraction ==> Jpeg Encode

* Hardware acceleration:

​		Video Decode by CPU or GPU ==> FPS filter frame extraction

​         ==> NV JPEG GPU hardware encode ==> Jpeg file write



A quick experience to try in colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabitMF/bmf/blob/master/bmf/example/video_frame_extraction/video_frame_extraction_acceleration.ipynb)



# Get Started

## 1. Environment Setup
*   install the BMF
*   make sure the GPU environment is ready



### 1-1 pip install BMF packages

To install a GPU supported version of BMF:

```bash
pip install -i https://test.pypi.org/simple/ BabitMF-GPU==0.0.2
```

### 1-2 verify the FFmpeg libraries is installed and version is correct

Part of feature in BMF framework utilizes the FFmpeg demuxer/muxer/codec and filter as the built-in modules for video processing. **If the capability of ffmpeg is needed,it's neccessary for users to install supported FFmpeg libraries before using BMF.**

**Optional step**

Install ffmpeg and related libraries. For this demo, we don't have to do this step, because ffmpeg libraries are already installed in the Google Colab environment.

```bash
sudo apt install ffmpeg
```


**BMF supports the FFmpeg verions 4.2 to 5.1**



## 2. BMF Video Frame Extraction Demo
### 2-1 Normal way of video frame extraction by BMF

we can set up the BMF processing pipeline (decoding->fps extraction->jpeg encoding) and run to show a normal way of video frame extraction by BMF.

```python
import bmf

input_video_path = "/content/big_bunny_10s_30fps.mp4"
output_path = "./simple_%03d.jpg"
graph = bmf.graph({'dump_graph':1})

video = graph.decode({
    "input_path": input_video_path,
}).fps(2)

(
    bmf.encode(
        video['video'],
        None,
        {
            "output_path": output_path,
            "format": "image2",
            "video_params": {
                "codec": "jpg",
            }
        }
    )
    .run()
)
```



### 2-2 Customized Hardware Accelerated Jpeg Encode BMF module

And in terms of gpu acceleration that a c++ jpeg hardware encode BMF module is implemented. A gpu acceleration video frame extraction BMF graph can be set up and run.

Please reference the source code in the directory.

#### 2-2-1 Develop a jpeg hardware encode module

The jpeg_encode.cpp is the source code of module and also a cmake file is needed in order to compile the module.
Make sure to set the path of BMF libs, headers and FFmpeg's in the cmake file.

#### 2-2-2 About FFmpeg environment
If the environment didn't include libavfilter and libavutil headers of FFmpeg, a extra install is needed.

```bash
apt install libavfilter-dev libavutil-dev
```

And we need to add a line in the header of FFmpeg 4.2 which not include a define for AV_CUDA_USE_PRIMARY_CONTEXT:

```c++
#define AV_CUDA_USE_PRIMARY_CONTEXT (1 << 0)
```

 In another way, user can install FFmpeg 4.4+ to solve the problem.

#### 2-2-3 Compile the module and run a gpu accelerated graph

By now we already have a customer C++ module for hardware jpeg encode, and we can also create a simple Python module to write the image data coming from the jpeg encoder into files by file_io.py in the directory.

And the final graph to run the hardware accelerated video frame extration is ready:

```bash
python video_frame_extraction.py
```

Then we can got the extracted GPU encoded jpeg images.

