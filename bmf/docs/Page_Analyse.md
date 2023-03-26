/** \page Analytics 视频分析

# 视频分析

视频分析主要分析视频画面，获取相关信息。比如人脸检测，OCR检测，图像超分，图像插帧。

## 痛点
当仅仅只是简单分析视频，获取一些简单的信息，比如人脸检测获取人脸位置的需求，可能简单地串联模块就可以了。但当需要较多的模块合并在一起，并需要做一些编码操作。举个例子，超分算法希望应用到视频转码中，由于当前的转码采用的ffmpeg的框架，如果想集成的话，其需要将超分算法写成ffmpeg filter（c++）并且需要重新编译一个ffmpeg版本。如果采用BMF，写一个超分的算法模块可以是python版本，也可以是c++版本，然后串联一个pipeline，就可以完成超分的整个流程。


## 如何使用BMF搭建视频分析
以下以超分简单举例

可以简单参考example/predict/

### 构建自定义的Module
详细可以参照 TODO 如何搭建自定义的Module

当前的超分模块是采用pytorch框架训练的，并转换到onnx，所以当前的超分模块是python Module

其数据处理都在process(task)里面，task内部有输入队列，以及输出队列，在当前场景下，其有一个输入队列存储着VideoFrame，其有一个输出队列预期是保存超分模块分析获得的超分图片，

其逻辑可以简单认为是 

1. 从task的input 队列获取输入图片
2. 分析图片并获取超分图片（该内容可以认为是验证模型的代码做一定的数据处理即可）
3. 将超分图片插入到输出队列。

详细代码可以参考example/predict/onnx_sr.py。

### 构建pipelines分析

```
(
	bmf.graph()
	    .decode({'input_path': "../files/img_s.mp4"})['video']
	    .module('onnx_sr', {"model_path": "v1.onnx"})
	    .encode(None, {"output_path": "../files/out.mp4"})
	    .run()
)
```
详细代码可以参考 example/predict/predict_sample.py
