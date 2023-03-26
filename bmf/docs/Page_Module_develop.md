# BMF模块开发（对外）
模块是BMF内部最小执行单位，每个模块完成视频处理的一个步骤，如解码，编码，分析等

目前BMF支持Python和C++,Go三种自定义模块

## 整体概述：
bmf engine采用动态加载的方式加载对应的module，根据一定的调度策略（用户指定），生成一些任务Task发送给各个BMF模块去处理，当处理完一个转码任务（一组task）的时候，（即module返回的task的timestamp被设置为Done的时候）框架会关闭BMF 模块（或者reset BMF模块）。

### BMF模块使用主要接口：

- 构造函数：
类初始化函数里可以进行模块的初始化操作，模块初始化参数使用option传入，option是Json形式的数据。
- 初始化模型（init）：
加载模型等比较耗时的操作
- 执行任务函数（process）：
任务的执行，
- 重置模块（reset）
用于在两组任务之间做一些清理的操作，在处理新的任务。
- 关闭模块（close）
模块关闭，执行资源释放等操作

### BMF的主要数据结构
#### 任务（Task）
整个graph的处理以task为基本单位，模块每次process处理一个task，task包含一个时间戳，以及若干个输入输出流（和模块输入输出个数相等），队列里面包含了本次process需要处理的数据包。

#### 包（Packet）
packet是graph里数据传输的存储单元。其主要有两个内容，timestamp，data

timestamp:数据包的时间戳，其中有两个特殊的时间戳需要特别注意。

- UNSET：无效的包，主要目的是
- BMF_EOF： 流结束标志，表明该路stream结束，不会有新的数据进来了。

data:数据内容。任意类型的数据。

#### BMF支持自动转换的数据结构。
bmf支持任意类型，如果在python，c++，Go不同的Module传递的话，目前仅支持VideoFrame，AudioFrame，BMFPacket。
##### 视频帧(VideoFrame)
视频帧是以ffmpeg的AVFrame做的一个封装，其存储着视频数据的相关信息。

c++ 结构：

python 结构：

go结构：
##### 音频帧(AudioFrame)
音频帧是以ffmpeg的AVFrame做的一个封装，其存储着音频数据的相关信息。

c++ 结构：

python 结构：

go结构：
##### 多媒体包(BMFPacket)
多媒体包是以ffmpeg的AVPacket做的一个封装，其存储着编码后的音视频数据。

c++ 结构：

python 结构：

go结构：

