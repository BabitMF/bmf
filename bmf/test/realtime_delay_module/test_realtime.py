import bmf
import sys
import os
import time # 导入 time 模块用于计时

# 定义输入和输出文件路径
input_file = "./test.mp4" # !!! 请确保在脚本同目录下有视频文件
output_file = "./realtime_output_simple_working.mp4" # 输出文件名

# 检查输入文件是否存在，如果不存在则退出并提示
if not os.path.exists(input_file):
    print(f"错误：找不到输入文件： {input_file}")
    print("请确保在脚本同目录下有一个名为 test.mp4 的视频文件。")
    sys.exit(1)

print(f"开始执行 BMF 图，输入文件： {input_file}")
print(f"输出文件将保存到： {output_file}")

try:
    # 创建一个 BMF 图对象
    graph = bmf.graph()

    # === 图节点和连接 ===

    # 1. 使用内置解码器解码输入文件
    # graph.decode() 返回一个字典，包含 'video' 和 'audio' 流（如果存在）
    decoded_streams = graph.decode({'input_path': input_file})

    # 获取解码器输出的视频流对象
    video_stream = decoded_streams['video']

    # 2. 将视频流输入到实时读取 Python 模块
    processed_video_stream = video_stream.module("realtime_delay_filter", option={})

    # 3. 将模块处理后的视频流输入给编码器并指定输出文件
    bmf.encode(
        processed_video_stream, # 模块输出的视频流
        None, # 没有音频流输入
        { # 编码器选项
            "output_path": output_file, # 指定输出文件路径
            "video_params": { # 视频编码参数
                "codec": "h264", # 使用 h264 编码
                "preset": "veryfast", # 编码速度预设（影响文件大小和编码速度）
                "crf": 23 # 质量控制参数
            },
            "format": "mp4", # 输出文件格式为 mp4
            "loglevel": "info" # 输出 BMF 日志信息
        }
    )

    # === 运行图并计时 ===
    print("运行 BMF 图...")
    start_time = time.time() # 记录开始时间
    graph.run() # 启动图的执行并等待直到完成
    end_time = time.time() # 记录结束时间
    execution_time = end_time - start_time # 计算执行时间

    # 如果运行到这里没有抛出异常，说明图执行成功
    print("BMF 图执行成功！")
    print(f"输出文件已保存到： {output_file}")
    print(f"BMF 图执行总耗时： {execution_time:.4f} 秒") # 打印执行时间，验证功能是否生效


# 捕获执行过程中可能发生的任何异常并打印错误信息
except Exception as e:
    print(f"\n--- BMF 图执行错误 ---")
    print(f"BMF 图执行过程中发生错误: {e}")
    print("请检查上面输出的详细错误信息和 BMF 的日志。")
    print("-----------------------------------")