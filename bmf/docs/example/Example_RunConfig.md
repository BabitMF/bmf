/** \page RunConfig 配置文件运行

BMF也支持配置文件方式构建graph：

```python
from bmf import GraphConfig

# build GraphConfig instance by config file
config = GraphConfig(file_path)
```

例子用的 \ref graph.json

配置后，可以直接执行graph：

```python
# create graph
my_graph = bmf.graph()

# run
my_graph.run_by_config(config)
```

如果需要完整代码，可以参考 \ref test_run_by_config.py
