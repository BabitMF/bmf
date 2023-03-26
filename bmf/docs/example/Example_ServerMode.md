/** \page ServerMode Server Mode

BMF也支持预构建模式。以下例子展示server初始化和使用：

```python
from bmf import ServerGateway

server_gateway = ServerGateway(graph)
server_gateway.init()
```

Server处理之前，先构建packet：

```python
packet1 = Packet()
packet1.set_timestamp(1)
video_info_list1 = [{'input_path': input_video_path_1}]
packet1.set_data({'type': InputType.VIDEO, 'input_path': video_info_list1})
```

之后产生线程，把packet发送到server graph：

```python
thread_ = threading.Thread(target=process_thread, args=(server_gateway, packet1))
thread_.start()
```


如果需要完整代码，可以参考 \ref test_server.py
