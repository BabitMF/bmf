/** \page PythonModule Python 模块

## 构建模块

模块构建如下：

```python
from bmf import Module

class my_module(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        pass

    def process(self, task):

        # process all input queues
        for (input_id, input_packets) in task.get_inputs().items():
            
            # processing code here

        return ProcessResult.OK
```

模块主要处理过程是在```process```。最终处理完毕，务虚返回处理结果。

如果需要完整模块代码，请参考 \ref my_module.py

## 使用模块

基本上模块的使用只需```module()```，输入模块名称就行：

**Python**
```python
bmf.graph()
    .module('my_module')
```
**C++**
```cpp
graph.Module({video["video"]}, "my_module", bmf::builder::Python, bmf_sdk::JsonParam());
```

如果需要完整应用代码，请参考 \ref test_customize_module.py (Python) 或 \ref c_module.cpp (C++)
