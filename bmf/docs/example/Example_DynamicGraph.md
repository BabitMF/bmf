/** \page DynamicGraph 动态Graph

### 相关接口：

dynamic_add()

dynamic_remove()

dynamic_reset()

update()

run_wo_block()

例子程序: \ref dynamical_graph.py

### 动态增加：

```python
main_graph = bmf.graph()
video1 = main_graph.decode({'input_path': input_video_path, 'alias': "decoder0"})

passthru = bmf.module([video1['video'], video1['audio']], 'pass_through',
           {
               'alias': "pass_through",
           },
           "", "", "immediate")
passthru.run_wo_block()
```
main_graph作为初始创建的graph，对于graph中的每个模块加入“alias”标记，用于后续动态增加关联使用。

初始graph运行的时候可以使用run_wo_block()是非阻塞的调用，也可以使用run()阻塞调用但需要启动另外的线程支持动态操作。

```python
update_decoder = bmf.graph()
video2 = update_decoder.decode(
                      {
                          'input_path': input_video_path2,
                          'alias': "decoder1"
                      })

outputs = {'alias': 'pass_through', 'streams': 2}
update_decoder.dynamic_add(video2, None, outputs)
main_graph.update(update_decoder)
```
动态增加节点需要明确要增加节点的输入流、输出流以及自身的配置，最后使用update()接口执行实际的操作。

### 动态删除：

```python
remove_graph = bmf.graph()
remove_graph.dynamic_remove({'alias': 'decoder1'})
main_graph.update(remove_graph)
```
动态删除只需要指定要删除节点的alias即可。

### 动态配置：

```python
main_graph.update(remove_graph)
option =  {       
          'alias': 'encode1',
          'output_path': output_path,
          'video_params': {
                'codec': 'h264',
                'width': 320,
                'height': 240,
                'crf': 23,
                'preset': 'veryfast'
           }
}
reset_graph = bmf.graph()
reset_graph.dynamic_reset(option)
main_graph.update(reset_graph)
```
动态配置只需要把需要配置的节点alias以及具体参数写成json格式的变量，作为dynamic_reset()的参数。

### 回调方式：

有些应用场景需要在某些模块节点中决定什么时候去动态的增删和配置，这种情况可以用BMF的回调机制去配合实现，详见例子程序的test_dynamical_graph_cb()