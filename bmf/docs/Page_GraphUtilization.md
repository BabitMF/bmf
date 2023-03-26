/** \page GraphUtilization GraphUtilization工具

# GraphUtilization Tool

BMF GraphUtilization（或 BMF Visualization 工具）是一个 BMF 可视化工具，目的是为了能够让用户能够更有效，更方便地可视化和分析 BMF 使用时所产生的日志信息。该工具涵盖的功能是：
- 框架所产生的 graph config - 显示构造的 graph，用户能够与可视图形进行交互，查看每个实体的信息
- [Trace 工具](#./Page_TraceTool_EN.md)所产生的 tracelog - 显示 Trace 工具所收集的数据，对数据进行统计

GraphUtilization 工具网址：http://
（以上网址是暂时的，迁移后再更新）

## Graph 构建图

GraphUtilization 的最基本特能是显示 graph - 除了 graph 可视化（左侧）之外，也是含有 JSON 编辑（右侧）。用户只需要加载 graph config JSON 就能看到所配置的 graph：

![Graph visualization](http://)

左侧是显示graph的功能，用户能够：
- 进行 pan, zoom 动作
- 右下角有个 mini map
- Draggable graph nodes
- Mouse hover 现实 node 信息

在右侧是 JSON 编辑：
- 显示 graph.json 的 JSON 内容
- 编辑 JSON，更新左侧 graph 的显示

## Trace 数据统计

GraphUtilization 也能对 tracelog.json 的数据进行统计：

![Trace info](http://)

![Charts display](http://)

## 使用工具

在右面板的顶部，有多个选项卡可在查看 JSON 或 Trace 信息之间切换。 加载相应的文件（对于 JSON 编辑选项卡为 graph.json，对于 Trace 统计选项卡为 tracelog）以显示信息。

GraphUtilization 无需同时加载 graph config 和 tracelog（可以仅加载 graph config，或可以仅加载 tracelog）。对于整体完整的可视化，建议从同一 graph 加载 graph.json 和 tracelog。
