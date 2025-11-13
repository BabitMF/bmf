# 实时延迟滤镜模块 (Python)

## 简介

本文档提供了关于 Babit 多媒体框架 (BMF) 中 `realtime_delay_filter` Python 模块的概述。此模块旨在模拟 FFmpeg 的 `-re`（以原始帧率读取输入）选项的行为，允许 BMF 图根据数据包的时间戳以实时速度处理多媒体流。

## 功能

`realtime_delay_filter` 模块作为一个直通（passthrough）滤镜，用于控制多媒体数据包的流速。它读取输入的包，并根据它们的显示时间戳（PTS）和处理开始以来的实际经过时间，引入延迟，以确保数据包的释放速度与时间戳所指示的实际播放速度同步。

这在以下场景中特别有用：

* 模拟实时流，其中数据必须以特定速率处理。
* 使用受实时限制的输入测试下游模块或系统。
* 分析多媒体流时关注时序和同步。

## 实现细节

* **语言：** Python
* **基类：** 继承自 `bmf.Module`。
* **核心逻辑：** 模块使用 `time.time()` 函数跟踪实际时间，并将其与输入数据包的微秒单位时间戳（`Packet.timestamp`）进行比较。如果一个包在其理论上的实时显示时间点（根据其时间戳相对于第一个包的时间戳和处理开始时间计算得出）之前到达，模块将使用 `time.sleep()` 暂停执行所需的时长。
* **时间戳单位：** 假设输入包的时间戳单位是微秒（这与许多 BMF 内置模块如 `c_ffmpeg_decoder` 的标准一致）。
* **数据包处理：** 模块从其输入队列读取包，应用实时延迟逻辑后，将原始包（不修改其内容）放入相应的输出队列。
* **流结束 (EOF)：** 正确处理 `bmf.Timestamp.EOF` 包，以向所有下游模块和 BMF 图通知流已结束。

## 使用方法

### 在 BMF 图中使用

在构建 BMF 图时，可以通过指定模块名称 `"realtime_delay_filter"` 来创建模块节点。

#### Python API 示例

以下是一个使用 `realtime_delay_filter` 模块的简单 BMF 图示例，用于以实时方式处理视频文件：

```python
import bmf
import sys
import os
import time

# 定义输入和输出文件路径
input_file = "../../files/test.mp4" # 请根据你的测试文件实际位置调整路径
output_file = "./realtime_output_with_filter.mp4"

# --- (可选) 输入文件检查 ---
# if not os.path.exists(input_file):
#     print(f"错误：找不到输入文件： {input_file}")
#     sys.exit(1)
# print(f"开始执行 BMF 图，输入文件： {input_file}")
# print(f"输出文件将保存到： {output_file}")
# --- 结束可选 ---

try:
    # 创建一个 BMF 图对象
    graph = bmf.graph()

    # 解码输入视频文件
    decoded_streams = graph.decode({'input_path': input_file})

    # 选择视频流
    video_stream = decoded_streams['video']

    # 通过 realtime_delay_filter 模块处理视频流
    # 模块名称必须与放置在 bmf/modules/ 目录下的文件名一致（不带 .py 后缀）
    processed_video_stream = video_stream.module("realtime_delay_filter", option={})

    # 编码处理后的视频流到输出文件
    bmf.encode(
        processed_video_stream, # 处理后的视频流
        None,                   # 在此简单示例中不处理音频流
        {
            "output_path": output_file,
            "video_params": {
                "codec": "h264",
                "preset": "veryfast", # 编码速度预设
                "crf": 23 # 质量控制参数
            },
            "format": "mp4", # 输出文件格式
            "loglevel": "info" # 设置日志级别
        }
    )

    # 运行图并测量执行时间
    print("正在运行带有 realtime_delay_filter 的 BMF 图...")
    start_time = time.time()
    graph.run()
    end_time = time.time()
    execution_time = end_time - start_time

    print("BMF 图执行完成。")
    print(f"输出文件已保存到： {output_file}")
    print(f"总执行时间： {execution_time:.4f} 秒")


except Exception as e:
    print(f"\n--- BMF 图执行错误 ---")
    print(f"BMF 图执行过程中发生错误: {e}")
    print("请检查详细错误信息和 BMF 日志。")
    print("-----------------------------------")
```

*注意：请调整示例中 `input_file` 的路径，使其正确指向你的测试视频文件（例如，相对于测试脚本位于 `../../files/test.mp4`）。*

## 选项

目前，`realtime_delay_filter` 模块不接受通过 BMF 图中的 `option` 参数进行配置的任何选项。它完全基于输入数据包的时间戳和系统时钟进行操作。

*（如果你将来添加选项，例如速度乘数，请在此处进行文档说明。）*

## 输入/输出流

* **输入：** 接受一个或多个输入流。它设计用于处理包含有效时间戳的数据包流（例如，来自解码器的视频或音频流）。
* **输出：** 为每个输入流产生相应的输出流，在应用实时延迟后直通原始数据包。输出流的数量和类型与输入流匹配。

## 注意事项与限制

* **时序精度：** 实时模拟依赖于 Python 的 `time.sleep()`，这可能无法提供高精度的延迟，特别是对于非常短的时间间隔。实际处理时间可能会由于系统调度和开销而略微超过理论时长。
* **时间戳单位：** 模块假定输入时间戳单位是微秒。请确保上游模块生成的时间戳单位正确，以实现准确的实时模拟。
* **多流同步：** 尽管模块处理来自多个输入流的数据包，但延迟是根据每个数据包的时间戳相对于开始时间独立应用的。此基本实现并未明确处理复杂的多流间同步细节（超出基于时间戳的简单 pacing）。
