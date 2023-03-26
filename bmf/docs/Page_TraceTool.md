/** \page TraceTool Trace工具

# Trace 工具

Trace 主要功能是吧一些重要事件记录起来，以便于进行故障排除或分析。Trace 在运行时记录下有关信息，然后在最后写入 tracelog 日志。

## 启动 Trace

Trace tool 可通过环境变量启动：
```bash
$ export BMF_TRACE=ENABLE
```
默认情况下，引擎中提供了多种 Trace 类型。启用 Trace 后允许在运行时记录这些事件。

为了向用户提供较低级别的控制，以最大程度地减少不必要的跟踪事件收集，用户可以选择仅启用选定的 [Trace 类型](#trace-类型)（以逗号分隔）：
```bash
$ export BMF_TRACE=PROCESSING,SCHEDULE
```

## 禁用 Trace

默认情况下，Trace 是禁用的。 但是，如果之前已设置了用于跟踪启用的环境变量，则可以将其禁用：
```bash
$ unset BMF_TRACE
```

## Step-by-Step Tutorial
- **[使用 Trace 工具](https://)** - Trace 工具的基本使用

## Trace 分析

工具在执行完毕后会打印和生成 tracelog 日志。用户可通过不同方式参考 Trace 所收集的信息。

### 1. Console 打印

使用 Trace 工具时，在 graph 执行完毕，会把 Trace 的一些信息打印出来：

![Trace printing](http://)

打印出的信息包含：
- **Runtime Information**
  Graph 总执行时间。如果有动用 [BMF_TRACE_INIT](#bmf_trace_init) 接口，也显示从 BMF_TRACE_INIT 到 graph 执行完毕的时间。这些时间不包括 tracelog 生成时间。
- **Events Frequency**
  每个 Trace event 的发生次数（降序排列）。
- **Events Duration**
  每个 duration event 的执行时间：
  - **Total**：总数
  - **Total (%)**：运行总执行时间占多少百分比
  - **Ave**：平均值
  - **Min, Max**：最小值和最大值
- **Queue Information**
  Stream 的队列信息：
  - **Limit**：队列大小上限（如果无限，则为零）
  - **Ave**：队列中的平均项目数
  - **Max**：队列中的最大项目数
- **Throughput**
  1秒内处理的平均 input packets
- **Trace Information**
  一些关于 Trace 工具的信息，如每个线程总共多少 Trace event，多少 event 因为 buffer 不足而 overflow，logging 时间等等。

默认情况下，Trace 信息将在 graph 执行结束时把 Trace 信息打印到 console。 但是，这将导致在代码执行结束之前增加处理时间。 如果有需要可以禁用打印：
```bash
# Default is ENABLE
$ export BMF_TRACE_PRINTING=DISABLE
```

### 2. Chrome Tracing

Tracelog JSON日志形式是符合 Chrome Tracing（Chrome网络浏览器 - chrome://tracing/ ） 中使用的格式，因此可以使用 Chrome Tracing 进行可视化：

![Viewing tracelog in Chrome Tracing](http://)

Instant event 和 duration event 的表示很容易区分：

![Instant event and duration event](http://)

在记录结束时，Trace 将在日志末尾附加相关信息（Trace info），以指示日志已完成：

![Trace info display](http://)

在底部面板可看到：
- **Title**：Trace event 名字
- **Category**：Trace type
- **Start**：Trace event 发生时间
- **Wall Duration**：Duration event 执行时间（instant event 没有这个参数）
- **Args**：Trace info 或 user info（附加参数）

注意：如果 overflow count 不为 0，则意味着分配的当前 buffer 不足，并且某些事件未记录（overflow）。 为了不错过任何跟踪事件，建议[增加缓冲区大小](#buffer-size)。

### 3. GraphUtilization 工具

除了 Chrome Tracing 之外，[BMF GraphUtilization 工具](./Page_GraphUtilization.md) 也支持跟踪事件的可视化：

![Visualizing tracelog using BMF GraphUtilization](http://)

GraphUtilization 工具能显示 graph 和一些 Chrome Tracing 无法显示的信息或图表。

## 例子

### 例子 1：预加载模式

作为使用 Trace 工具来识别实现中的瓶颈的示例，请考虑一个典型的代码转换示例：

![Typical transcoding DAG](http://)

注意: 以上 graph 的构建图可通过 [GraphUtilization 工具](./Page_GraphUtilization.md) 显示

使用以下 Python 代码：

```python
module_name = "analysis"

(
    bmf.graph({"dump_graph": 1})
        .decode({'input_path': input_video_path})['video']
        .scale(320, 240)
        .module(module_name, {
            "name": "analysis_SR",
            "para": "analysis_SR"
        })
        .encode(None, {
        "output_path": output_path,
        "video_params": {
            "width": 300,
            "height": 200,
        }
    }).run()
)
```

如果没有使用预加载模式（pre-module）方式，执行这个 graph 是挺耗时的。利用 Trace 工具所生成的 tracelog 可以在 Chrome Tracing 看出 Node 2（也就是以上图的 analysis 模块 - 括号内的数字表示 Node ID）的 initialisation 耗3秒：

![Module initialization time](http://)

使用预加载模式（pre-module）优化后的代码：

```python
module_name = "analysis"

pre_module = bmf.create_module(module_name, {
    "name": "analysis_SR",
    "para": "analysis_SR"
})

(
    bmf.graph({"dump_graph": 1})
        .decode({'input_path': input_video_path})['video']
        .scale(320, 240)
        .module(module_name, option, pre_module=pre_module)
        .encode(None, {
        "output_path": output_path,
        "video_params": {
            "width": 300,
            "height": 200,
        }
    }).run()
)
```

预加载模式（pre-module）能够很明显的减短 graph 执行时间：

![Analysis module in pre-module mode](http://)

### 例子 2：排查 Hang 问题

在写模块时，有时会碰到一些 bug 或瓶颈，Trace 工具能够帮助用户寻找问题的来源。

以下例子是发生在 encoder 模块，执行时 graph 似乎挂起（hang）。由于 graph 没执行完毕，tracelog 没有生成。但是，使用 [trace_format_log](#trace_format_log) 后会从构建 tracelog，能在 Chrome Tracing 上分析：

![Hang problem troubleshoot](http://)

基本的 Trace 节点虽然有限，但是能看得出哪个 node（以及模块）没有完成 process_node。如果需要深一步的分析，可以从函数里添加 CUSTOM Trace 节点（用 Trace 接口），重新运行后能进一步的看到 process 函数的进程调用：

![Hang problem troubleshoot](http://)

在这个例子里可看到，每当有一个新的 video frame，handle_video_frame 函数需要处理一千多个 sync frames，造成执行时间很长（看起来好像 hang）。

---

## Trace 类型

目前有几种 Trace 类型的可用：

| Type Name | Description |
| --------- | ----------- |
| PROCESSING | 一种 duration event，用于捕获节点和模块的处理持续时间：<br />**Node**: init, process_node 函数 <br />**Module**: init, process 函数 |
| SCHEDULE | 一种 duration event，在计划安排执行某个节点时发生。除了每个节点 schedule 以外，还有 THREAD_X_WAIT（线程等待 queue 有数据或 terminating state） |
| QUEUE_INFO | 一种即时事件，包含有关各个流的状态（在该时间点排队的数据包数量）的信息 |
| THROUGHPUT | 一种即时事件，计算节点的吞吐量 |
| CUSTOM | 用户可以使用此 Trace 类型来记录自己的事件 |

注意: **duration event** 是一种类型的事件，具有开始时间和结束时间，**instant event** 是瞬时的事件。

## 使用 Trace 接口

**目前 Trace 接口只能在 C++ 和 Python 使用。**

如果用户需要自己添加监控点，可以用SDK里的Trace接口：
```
BMF_TRACE({Trace Type}, {Event Name}, {Event Phase})
```

最终的事件名称由 ```Event Name``` 本身以及后续的 ```subname```（通常是称为跟踪接口的函数的名称）组成：```EventName:FunctionName```

```Event Phase``` 取决于事件的类型：是 instant event（仅是 ```NONE```），还是 duration event 具有 ```START``` 和 ```END```。

### 基本使用

用户可以通 ```CUSTOM``` 的 Trace 类型自由调用此类跟踪类型：

**C++**
```c++
#include "bmf/sdk/cpp_sdk/include/trace.h"

BMF_TRACE(CUSTOM, "Test Name");

// or

BMF_TRACE(CUSTOM, "Test Name", NONE);
```

**Python**
```python
from bmf import BMF_TRACE, TraceType, TracePhase

# Create an instant event
BMF_TRACE(TraceType.CUSTOM, "Test Name", TracePhase.NONE)
```

对于 duration event:

**C++**
```c++
#include "bmf/sdk/cpp_sdk/include/trace.h"

// Create a duration event (with START and END)
BMF_TRACE(CUSTOM, "Test Name", START);

// Some processing here

BMF_TRACE(CUSTOM, "Test Name", END);
```

**Python**
```python
from bmf import BMF_TRACE, TraceType, TracePhase

# Create a duration event (with START and END)
BMF_TRACE(TraceType.CUSTOM, "Test Name", TracePhase.START)

# Some processing here

BMF_TRACE(TraceType.CUSTOM, "Test Name", TracePhase.END)
```

### 附加参数

如果用户希望在 Trace event 中包含其他信息（user info），则可以使用另一个接口：

**C++**
```c++
#include "bmf/sdk/cpp_sdk/include/trace.h"

// Create user info
TraceUserInfo info = TraceUserInfo();
info.set("key1", "string_value");   // string value
info.set("key2", 8);                // int value
info.set("key3", 0.99);             // double value

// Create an instant event
BMF_TRACE_INFO(CUSTOM, "Test Name", NONE, info);
```

**Python**
```python
from bmf import BMF_TRACE_INFO, TraceType, TracePhase, TraceInfo

# Create user info
info = TraceInfo()
info.set("key1", "string_value")    # string value
info.set("key2", 8)                 # int value
info.set("key3", 0.99)              # float value

# Create an instant event
BMF_TRACE_INFO(TraceType.CUSTOM, "Test Name", TracePhase.NONE, info)
```

### BMF_TRACE_INIT

如果在执行 graph 之前想要需要加一个 start event（可以测量 graph 执行之前的总执行时间，包括 builder 层），可以动用这个接口：

**C++**
```c++
#include "bmf/sdk/cpp_sdk/include/trace.h"

BMF_TRACE_INIT();

// Graph builder

graph.start();
```

**Python**
```python
from bmf import BMF_TRACE_INIT

BMF_TRACE_INIT()

# Graph builder

graph.run()
```

---

## Trace 配置

根据应用程序要求，可以通过环境变量配置 Trace 设定。

### Buffer Size

如果用户需要改 Trace buffer size（默认 1024），在使用BMF之前可以通过 environment variable 更改：
```bash
# Default is 1024
$ export BMF_TRACE_BUFFER_SIZE=2000
```

### Buffer Count

目前 Trace buffer次数是固定。用户也能自定：
```bash
# Default is dependent on std::thread::hardware_concurrency
# whish is the number of concurrent threads supported by the implementation
$ export BMF_TRACE_BUFFER_COUNT=2
```

---

## 日志收集

Trace event 被存储到 buffer 中，然后直接写入多个二进制日志文件（名称以 **log_XX.txt**），然后在 graph 关闭后再合并并写入最终的 tracelog（名称为 **tracelog_yyyymmdd_HHMMSS.json**）。对于 server 模式，例外情况是在处理每个任务之后都会生成 tracelog。生成 tracelog 后，二进制日志将自动删除。

**Tracelog 可以在与用户脚本执行的目录中找到**

生成 tracelog 的过程是挺耗时的：

![Logging time](http://)

如果用户需要跳过 tracelog 生成步骤，可以通过环境变量：

```bash
# Default is ENABLE
$ export BMF_TRACE_LOGGING=DISABLE
```

Trace 就不会生成 tracelog 文件，并且仍将保留二进制日志，同时也禁止 printing。虽然 tracelog 无法生成，但是二进制日志还是存在。

### trace_format_log

用户能手动将二进制日志文件转换为格式化的 tracelog，可以在包含二进制日志的同一目录中运行 ```trace_format_log``` 工具：

```bash
$ cd /path/to/binary/logs
$ trace_format_log
```
