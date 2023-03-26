/** \page Installation 安装部署

# 安装部署

BMF安装有三种方法，可以随着开发需求选择最适合的方法:
1. [镜像环境搭建](#镜像环境搭建)
2. [SCM包方式 (Linux)](#scm包方式)
3. [Pip Install 方式 (Linux/Mac)](#pip-install-方式)

---

## 镜像环境搭建

镜像环境的搭建更为简单，只需要拉取包含BMF环境的基础镜像，就可以进行开发。

**先决条件:**
在开发机上[安装Docker](https://docs.docker.com/install/)

### i. 镜像下载
```bash
$ docker pull <URL>
```

请参考下表，可以从各镜像中获取最新版本容器地址，将```<URL>```替换为所需要的镜像版本容器地址。

| 环境 | 请参阅下面的ICM页面以获取最新的容器地址
| --- | ---
| Python 3.7 | https://

举个例子，如果需要 BMF v1.0.0 Python 3.7版本：
```bash
$ docker pull 
```

### ii. 镜像运行

```bash
$ docker run --name <name> -it <URL> /bin/bash
```
提示：可以把```<name>```替换成想要的容器名称。

如果需要与系统共享文件，请改用以下：
```bash
$ docker run -it --name <name> -v <local path>:<container path> <URL> /bin/bash
```

*提示：将```<local path>```替换为开发机系统路径地址，而```<container path>```是容器路径地址。举个例子，```docker run -it --name <name> -v /Users/bytedance/Project:/workspace <URL> /bin/bash```*

---

## SCM包方式

当用户希望使用自己的环境，其可以拉取BMF的SCM包，然后构建使用。

### i. 拉取SCM包

在以下可以获取最新版本SCM地址：

| 目标环境 | 请参考以下SCM页面以获取最新版本
| --- | --- | ---
| Debian 10 Python 3 | https://

```bash
$ wget -q <URL> -O \
    bmf.tar.gz && mkdir bmf && \
    tar -zxvf bmf.tar.gz -C bmf && \
    cp -r bmf/bmf <packages-path>/bmf &&\
    cp -r bmf/3rd_party <packages-path>/bmf/3rd_party && \
    rm -rf bmf.tar.gz && rm -rf bmf
```

请按照下面的相应环境进行安装。把```<URL>```更换成最新版本号，如 v1.0.0: http://
也把```<packages-path>```改成Python packages安装地址，例如```/usr/local/lib/python3.7/dist-packages/```。

### ii. 设置环境变量

请设置以下environment variables和保留在```~/.bashrc```:
```bash
$ cd <packages-path>/bmf
$ echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rd_party/ffmpeg:$(pwd)/lib:$(pwd)/3rd_party/lib" >> ~/.bashrc
$ echo "export PYTHONPATH=$PYTHONPATH:$(pwd)/3rd_party:$(pwd)/lib:$(pwd)" >> ~/.bashrc
$ echo "export PATH=$PATH:$(pwd)/3rd_party/ffmpeg:$(pwd)/bin" >> ~/.bashrc
$ echo "export LIBRARY_PATH=$LIBRARY_PATH:$(pwd)/3rd_party/ffmpeg:$(pwd)/lib:$(pwd)/3rd_party/lib" >> ~/.bashrc
$ echo "export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$(pwd)/include:$(pwd)/3rd_party/include" >> ~/.bashrc
$ echo "export C_INCLUDE_PATH=$C_INCLUDE_PATH:$(pwd)/include:$(pwd)/3rd_party/include" >> ~/.bashrc
$ source ~/.bashrc
```

---

## Pip Install 方式

**目前只支持 Debian 10 / Mac 10.15，Python 3.6 - 3.9**

BMF可以通过内部PyPI安装：
```bash
$ pip install byted-bmf
```

注意：为了访问私有 PyPI，将 index URL 设置为 *https://，或：*pip3 install -i https://

---

## 常见问题

在安装BMF后运行时遇到问题，请参考文档：[BMF 运行环境常见问题](https://)
