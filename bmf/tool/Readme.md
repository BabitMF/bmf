# BMF tool 的使用

### 1、内存泄漏脚本的使用

使用方式：

cd bmf/tool

python3 test_mem_leak.py

说明：

如需增加内存泄漏检测案例，测试代码需在bmf/mem_leak_test中添加，
并在bmf/tool/mem_leak_case.json中添加案例的完整路径。
生成的内存泄漏日志文件均位于bmf/tool/mem_leak_log目录下。

### 2、CMD命令行运行工具的使用

简单使用可见 bmf.sh 中的文字说明

设计详情请看 [CMD 运行 BMF](https://)
