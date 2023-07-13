## BMF Data Convert Backend
Currently, the backend interface is under testing.

### Background
An all-in-one solution is needed when multiple dimontion factors involved in video process pipeline such as CPU/GPU devices, YUV420/NV12 or RGB24/RGB48, and AVFrame or Torch structure.
As a framework, each of module just focus on it's own target and data requirement, but it becomes complex when multiple modules work together as below supper resolution pipeline:
<img src="./images/backend.png" style="zoom:30%;" />

We can see that different modules have their own data requirement, the decode module output FFmpeg AVFrame with YUV420 pixel format which located on CPU memory, while the Trt SR module requires input data to be torch with RGB24 which located on GPU memory, after hardware accelerated SR by the Trt module, the output data need to be encoded by GPU, so the HW encode module can get AVFrame with NV12 pixel format which located on GPU memory to encode it by GPU.

It tends to include the capabilities of video data convertion below:
- pixel format and color space
- devices between CPU and GPU
- different media types such as avframe, cvmat and torch

### C++ Interface

### Python Interface