/** \page Subgraph Subgraph

如果需要使用subgraph，可以参考以下例子。

## 构建 Subgraph

若需要仔细参考subgraph的构建，请看 \ref subgraph_module.py 。这个例子先把原来的video做vertical flip，然后在overlay一个image。

```python
from bmf import SubGraph

class subgraph_module(SubGraph):
    def create_graph(self, option=None):
        # Build subgraph here
```

构建subgraph时，需要把input streams名称输入Subgraph的```self.inputs```：

```python
# input stream name, used to fill packet in
self.inputs.append('video')
self.inputs.append('overlay')
```

之后，需要把output streams输出：
```python
# finish creating graph
self.output_streams = self.finish_create_graph([output_stream])
```

## 使用 Subgraph

用subgraph的方法跟其他module相似：

```python
bmf.module([video['video'], overlay['video']], 'subgraph_module')
```

如果需要完整代码，可以参考 \ref test_subgraph.py
