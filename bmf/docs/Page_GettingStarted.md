/** \page GettingStarted Getting Started

## Brief Introduction

ByteMF（ByteDance Multi-Media Framework，简称BMF或ByteMF）是一套通用视频处理框架。

BMF能够提供简洁易用的接口、灵活的调度和扩展性，用模块化的方式动态扩展、管理和复用视频处理的原子能力。

通过BMF，用户可以达到以下目标：

<img src="./images/goals.png" style="zoom:30%;" />





通过BMF，用户可以根据需求，快速高效的构建或简洁或复杂的应用，下面第一幅图的BMF Graph中展示了一个简单的音视频转码处理，其中包括三个Module，之间通过Stream相关联，将数据流以Packet为载体传递，每个Module可以指定独立的Thread被BMF引擎调度运行。

<img src="./images/simple_graph.png" style="zoom:30%;" />

而第二幅图展示了一个较为复杂的音视频拼接与overlay的Graph。

<img src="./images/complex_graph.png" style="zoom:40%;" />



BMF中的一些基本概念如下：

- Graph：用户可以通过接口层开发、通过BMF构建的DAG，往往是可以运行的pipeline、应用。

- Module/Node: 用户可以使用BMF内置的或者通过模块SDK开发的模块。BMF把每个视频处理步骤都看作是一个原子能力，而它的载体就是模块，模块支持多语言动态扩展、管理和复用。Node是实例化的Module。

- Stream：使用BMF接口层在构建Graph，大多时候都是使用Stream的对象去连接。数据在BMF框架中传递的通路就可以看作是Stream流，可以是任意数据形式的输入输出通路。

- Packet: 数据在BMF中的载体。任意的数据类型包括VideoFrame，AudioFrame会被封装成为BMF中的Packet在模块之间传递。

- Task: Task是BMF引擎的调度单元，引擎层针对不同的Node生成Task调度到需要处理的模块中，模块的process()入口获取到Task进行处理。



更多BMF的复杂特性以及使用场景，详见 \ref BasicConcepts ，\ref Solutions 以及 \ref Examples 。



## The First Program

### 运行环境

本例子直接下载BMF发布的镜像作为运行和开发环境，其他安装部署方法详见：安装部署。

`$ docker pull `

`$ docker run --name bmf_dev  `

### 程序

下面hello_word.py程序通过BMF应用层接口：graph(), decode(), encode(), run()，创建运行一个视频转码的pipeline：demux + decode -> encode + mux

```python
def hello_world():
    input_video_path = "/opt/tiger/bmf/bmf/files/img.mp4"
    output_path = "./simple.mp4"

    # 创建BMF Graph
    graph = bmf.graph()

    # 构建解码流
    streams = graph.decode({
        "input_path": input_video_path
    })

    (   # 将前面的streams流分解为video和audio作为encode接口输入，以及编码参数，创建编码流并且直接运行
        bmf.encode(
            streams['video'],
            streams['audio'],
            {
                "output_path": output_path,
                "video_params": {
                    "codec": "h264",
                    "width": 320,
                    "height": 240,
                    "crf": 23,
                    "preset": "veryfast"
                },
                "audio_params": {
                    "codec": "aac",
                    "bit_rate": 128000,
                    "sample_rate": 44100,
                    "channels": 2
                }
            }
        )
        .run()
    )
    
if __name__ == '__main__':
    hello_world()
```
### 运行

`$ python3 hello_world.py`


