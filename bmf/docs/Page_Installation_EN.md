# Installation

There are multiple ways to get started with BMF:
1. [Run BMF Container](#run-bmf-container)
2. [Download SCM Repository (Linux)](#scm-installation)
3. [Pip Install (Linux/Mac)](#pip-install)

---

## Run BMF Container

BMF Docker images can be downloaded directly for use. This is the most straightforward installation method.

**Prerequisites:**
[Install Docker](https://docs.docker.com/install/) on your local host machine.

### i. Download image
```bash
$ docker pull <URL>
```
Replace ```<URL>``` with the image URL. Refer to the following table for the latest image source.

| Environment | Refer to the following ICM page for latest image URL
| --- | ---
| Python 3.7 | https://

E.g. To obtain an image for BMF v0.1.0 Python 3.7:
```bash
$ docker pull 
```

### ii. Running the Docker container
Replace ```<name>``` with the desired container name.
```bash
$ docker run --name <name> -it <URL> /bin/bash
```

If file sharing between local host and the docker container is required, use the following instead:
```bash
$ docker run -it --name <name> -v <local path>:<container path> <URL> /bin/bash
```

*Tips: Substitute ```<local path>``` with the path of shared folder to map onto the Docker container's ```<container path>```. E.g. ```docker run -it --name <name> -v /Users/bytedance/Project:/workspace <URL> /bin/bash```*

---

## SCM Installation

For development on existing environment, download and install directly from the SCM package.

### i. Download and extract

The latest version can be obtained from the SCM page of each environment:

| Target Environment | Refer to the following SCM page for latest version
| --- | --- | ---
| Debian 10 Python 3.7 | https://

```bash
$ wget -q <URL> -O \
    bmf.tar.gz && mkdir bmf && \
    tar -zxvf bmf.tar.gz -C bmf && \
    cp -r bmf/bmf <packages-path>/bmf &&\
    cp -r bmf/3rd_party <packages-path>/bmf/3rd_party && \
    rm -rf bmf.tar.gz && rm -rf bmf
```

Replace ```<URL>``` with the latest version number e.g. for v0.1.0: ```http://```.

Change ```<packages-path>``` to the path of Python packages directory e.g. ```/usr/local/lib/python3.7/dist-packages/```.

### ii. Setting environment variables

The following environment variables are required to be set and persisted i.e. set in ```~/.bashrc```:
```bash
$ cd <packages-path>/bmf
$ echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rd_party/ffmpeg:$(pwd)/lib:$(pwd)/3rd_party/lib" >> ~/.bashrc
$ echo "export PYTHONPATH=$PYTHONPATH:$(pwd)/3rd_party:$(pwd)/lib:$(pwd)" >> ~/.bashrc
$ echo "export PATH=$PATH:$(pwd)/3rd_party/ffmpeg:$(pwd)/bin" >> ~/.bashrc
$ echo "export LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc
$ echo "export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$(pwd)/include:$(pwd)/3rd_party/include" >> ~/.bashrc
$ echo "export C_INCLUDE_PATH=$C_INCLUDE_PATH:$(pwd)/include:$(pwd)/3rd_party/include" >> ~/.bashrc
$ source ~/.bashrc
```

---

## Pip Install
Currently this is precompiled for Debian 10 / Mac 10.15, Python 3.6 - 3.9 only

BMF can be installed directly from private PyPI:
```bash
$ pip3 install byted-bmf
```

Note: In order to access private PyPI, set the base URL of the index to *https://, or use: *pip3 install -i https://
