/** \page PreModule 预加载模式

这个例子里，用了一个 \ref analysis.py

当应用场景需要预加载模式，先初始化：

**Python**
```python
pre_module = bmf.create_module(module_name, option)
```
**C++**
```cpp
nlohmann::json pre_module_option = {
    {"name", "analysis_SR"},
    {"para", "analysis_SR"}
};
auto pre_module = bmf::builder::GetModuleInstance("analysis", pre_module_option.dump());
```

之后就能直接使用：

**Python**
```python
bmf.graph()
    .module(module_name, option, pre_module=pre_module)
```
**C++**
```cpp
auto analyzed = output.PythonModule({}, "analysis", bmf_sdk::JsonParam());
analyzed.SetPreModule(pre_module);
```

如果需要完整代码，可以参考 \ref test_pre_module.py (Python) 或 \ref c_mode.cpp (C++)
