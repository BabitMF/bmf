/** \page Module C++ 模块

## 构建模块

模块构建如下：

```c++
#ifndef BMF_COPY_MODULE_H
#define BMF_COPY_MODULE_H

#include "module.h"

class CopyModule : public Module
{
public:
    CopyModule(int node_id, JsonParam option) : Module(node_id,option) { }

    ~CopyModule() { }

    virtual int process(Task &task);
};

#endif
```

模块主要处理过程是在```process```。最终处理完毕后，需返回处理结果：

```c++
int CopyModule::process(Task &task) {
    PacketQueueMap &input_queue_map = task.get_inputs();
    PacketQueueMap::iterator it;

    // process all input queues
    for (it = input_queue_map.begin(); it != input_queue_map.end(); it++) {
        
        // processing code here
    }

    return 0;
```

请注意，模块需要注册：

```c++
REGISTER_MODULE_CLASS(CopyModule)
```

如果需要完整模块代码，请参考：
- \ref copy_module.h
- \ref copy_module.cc

## 使用模块

要使用C++模块，编译后在```c_module()```输入模块名称，```c_module_path```和```c_module_entry```入口：

**Python**
```python
c_module_path = './libcopy_module.so'
c_module_entry = 'copy_module:CopyModule'

# c module processing
video_2 = (
    video['video'].c_module("copy_module", c_module_path, c_module_entry)
)
```
**C++**
```cpp
auto video_2 = graph.Module(
    {video["video"]}, 
    "copy_module", 
    bmf::builder::CPP, 
    bmf_sdk::JsonParam(), 
    "CopyModule", 
    "./libcopy_module.so", 
    "copy_module:CopyModule"
);
```

如果需要完整应用代码，请参考 \ref test_customize_module.py (Python) 或 \ref c_module.cpp (C++)
