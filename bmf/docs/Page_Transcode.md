/** \page Transcode 转码


# 转码服务
转码服务既要满足业务需求，又要优化视频生产和消费体验，还要兼顾成本，提升转码服务的灵活性是非常重要且紧迫的一件事情。

## 当前转码服务架构图
整个转码服务是一个系统化工程，有用户层面的交互，模版以及工作流的视频生产，计算平台的资源支持，BMF主要参与工作流中转码函数的pipeline的构建与运行调度。

<img src="./images/transcode_server_architectur.png" style="zoom:40%;" />

## 转码函数
转码函数主要完成的是视频的加工。整个相关的如下：

<img src="./images/transcode_architecture.png" style="zoom:40%;" />

其主要有以下几个内容：

1. 转码源数据的获取，根据VID获取转码原视频。


2. 转码pipeline的搭建与执行。


3. 转码结果的上传，将转码获得的转码视频上传到VDA。


<img src="./images/transcode_graph.png" style="zoom:50%;" />

如以上图所示，通过BMF框架采用流式构建一个复杂的pipeline，其主要包括多个视频的解码，个性化的filter，多个流合并叠加，视频的编码。
