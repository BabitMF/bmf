# How to use NVIDIA GPU to accelerate video processing

NVIDIA GPUs have one or more hardware-based decoder and encoders which provides fully-accelerated hardware-based video decoding and encoding for several popular codecs.

Nowadays, many video processings rely on the deep learning. The deep learning models usually run on the NVIDIA GPUs and libraries developed by NVIDIA. So, transferring the decoding and encoding from CPU to GPU can obtain benifit in such cases. One obvious benefit is that we can reduce the copy overhead between CPU and GPU.

In the `gpu_transcode` folder, we provide various examples to show how to use GPU decoding and encoding as well as how to combine the FFmpeg CUDA filters in the BMF.

The examples are list below, we will explain them in detail.

- Decode videos
- Decode videos using multiple threads
- Encode videos
- Encode videos using multiple threads
- Transcode
- Transcode 1 to n
- Transcode using multiple threads
- Transcode with scale cuda filter
- Transcode with hwupload filter
- Transcode with scale npp filter
- Transcode with yadif filter
- Transcode with overlay cuda filter

## Decode

In the BMF, enabling GPU decoding is really simple. What you need to do is to add `"hwaccel": "cuda"` in the `"video_params"`.

You should note that if you use GPU to decode videos, the decoded frames are in the GPU memory. So if you want to manipulate at the cpu side, don't forget copy these frames into cpu memory. In the BMF, you can set GPU decoding followed by a `cpu_gpu_trans_module` or followed by a `hwdownload` filter.

See more details in the `test_gpu_decode()`.

## Encode

In the BMF, you can add `"codec": "h264_nvenc"` or `"codec": "h264_hevc"` in the encode module's `video_params` to enable GPU encoding. If the inputs of the encoder are in the GPU memory, you should add `"pix_fmt": "cuda"` to the `video_params`.

See more details in the `test_gpu_encode()` and `test_gpu_transcode()`.

## Transcode

For GPU transcoding, you should combine the GPU encoding and GPU encoding metioned before. Since all the intermediate data are in the GPU memory, we don't need to consider extra copying any more.

See more details in the `test_gpu_transcode()`.

`test_gpu_transcode_1_to_n()` shows the BMF can transcode one video to several videos in the same time. Just add more encode module with different parameters after the same decode module.

## Transcode with FFmpeg CUDA filters

There're many CUDA filters in the FFmepg that can be used in the BMF through `ff_filter`. Using CUDA filters eliminates the copy overhead exits in the CPU filters when we are using GPU transcoding.

Using these filters is really simple. just pass filter'name and paramters to the `ff_filter`. But you should be careful about where the data reserves. For example, in the `test_gpu_transcode_with_overlay_cuda()`, the logo is png and is decoded and processed in the CPU. The video is decoded by the GPU so the frames are in the GPU. Because we will use CUDA filters and GPU encoding, we should upload the result of logo to the GPU. Here we use hwupload_cuda filter.

## Multiple threads and multiple processes

Some GPUs may have more than one hardware-based decoders and encoders. In order to fully utilize these hardwares, we have to start as many instances as possible. BMF can launch these instances through python multi-threading and multi-processing. You can see the examples in the `test_gpu_decode_multi_thread_perf`, `test_gpu_encode_multi_thread_perf` and `test_gpu_transcode_multi_thread_perf`.

For multi-processing, there's one special thing we should notice so that we use a seperate script `test_gpu_decode_multi_processes.py` to show how to do it in the BMF. 

We should `import bmf` in the task function rather than at the beginning of the script file. 

## TensorRT Inference

For video processing that use deep learning models, they can use [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) to accelerate inference. TensorRT is an SDK for high-performance deep learning inference, includes a deep learning inference optimizer and runtime that delivers low latency and high throuphput for inference applications.

We provide two examples to show how to use TensorRT in BMF. One is the face detection, you can find it in the `examples/face_detect` folder. Another is the super resolution, it locates in the `examples/predict` folder.

### Build a TensorRT engine

Before you use TensorRT, you should build an engine from a trained model. There're many ways to do it. You can see more details from the official [document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html). In our examples, we introduce our commands to build the two engines.

For face detection example:
```
trtexec --onnx=version-RFB-640.engine --buildOnly --saveEngine=version-RFB-640.engine
```

For super resolution example:
```
trtexec --onnx=v1.onnx --minShapes=input:0:1x360x640x21 --optShapes=input:0:1x360x640x21 --maxShapes=input:0:1x360x640x21 --buildOnly --fp16 --saveEngine=v1.engine
```

### Write a TensorRT module

In order to write a TensorRT module, you have to prepare these things:

- engine path
- inputs shapes
- outputs buffer

The `engine path` and `inputs shapes` can be passed by the users. Once the input shapes are set, TensorRT can infer the outputs shapes automatically. So `outputs buffer` can be allocated without users' control. The steps are usually as follows:

1. Get the total number of inputs and outputs

```python
self.num_io_tensors_ = self.engine_.num_io_tensors
self.tensor_names_ = [self.engine_.get_tensor_name(i) for i in range(self.num_io_tensors_)]
self.num_inputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                        .count(trt.TensorIOMode.INPUT)
assert self.num_inputs_ == len(self.input_shapes_.keys()), "The number of input_shapes doesn't match the number of model's inputs."
self.num_outputs_ = [self.engine_.get_tensor_mode(self.tensor_names_[i]) for i in range(self.num_io_tensors_)] \
                        .count(trt.TensorIOMode.OUTPUT)
```

2. For each input, set its shape
```python
for i in range(self.num_inputs_):
    self.context_.set_input_shape(self.tensor_names_[0], self.input_shapes_[self.tensor_names_[0]])
```

3. Allocate the output buffer
```python
self.output_dict_ = dict()
for i in range(self.num_inputs_, self.num_io_tensors_):
    self.output_dict_[self.tensor_names_[i]] = mp.empty(self.context_.get_tensor_shape(self.tensor_names_[i]),
                                                        device=mp.kCUDA,
                                                        dtype=self.to_scalar_types(self.engine_.get_tensor_dtype(self.tensor_names_[i])))
```

The inputs of TensorRT are usually from the decoded frames. If we need do some image preprocessing, we can convert the frames to the torch tensors. So, we can use torch operations to do preprocessing. During the inference, we should pass set the pointer bindings of inputs and outputs. Both torch tensor and bmf tensor can obtained raw pointers through `data_ptr()`.

```python
for i in range(self.num_inputs_):
    self.context_.set_tensor_address(self.tensor_names_[i], int(input_tensor.data_ptr()))

for i in range(self.num_inputs_, self.num_io_tensors_):
    self.context_.set_tensor_address(self.tensor_names_[i], int(self.output_dict_[self.tensor_names_[i]].data_ptr()))
```

After setting the input/output bindings, we can start TensorRT execution by:
```python
self.context_.execute_async_v3(self.stream_.handle())
```

The outputs are BMF tensor since we create output buffer using `mp.empty`. If you want to process these outputs, you can convert these BMF tensors to Torch tensors.
```python
output_tensor = torch.from_dlpack(self.output_dict_[self.tensor_names_[-1]])
```
