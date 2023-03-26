/** \page Logging 日志

## BMF Logging

BMF 分别有 5个 log severity levels：


| Log Severity Levels |
|---|
| DEBUG |
| INFO |
| WARNING |
| ERROR |
| FATAL |

## 开关显示

如果需要在某个日志级别显示，可以设环境变量，显示首选log level：

```
# 显示 INFO 以上的信息
export BMF_LOG_LEVEL=INFO
```

但如果设了之后需要关掉显示，必须去掉有关环境变量：

```
BMF_LOG_LEVEL=DISABLE
```

## 日志接口

要记录消息，请使用 BMFLOG macro (C++) 或 Log.log 函数 (Python)：

**C++**
```
#include "log.h"

BMFLOG(BMF_INFO) << "Information here";
```

**Python**
```
from bmf import Log, LogLevel

Log.log(LogLevel.INFO, "Information here")
```

如果需要包括node id信息（比如在自己模块里添加日志功能）：

**C++**
```
#include "log.h"

BMFLOG_NODE(BMF_ERROR, node_id_) << "Error here";
```

**Python**
```
from bmf import Log, LogLevel

Log.log_node(LogLevel.ERROR, self.node_, "Error here")
```