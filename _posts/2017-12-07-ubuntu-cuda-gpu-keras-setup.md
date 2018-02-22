---
layout: post
title: Setting up a Keras / tensorflow environment in a vanilla Ubuntu
last_modified_at: 2018-01-22T16:00:00+01:00
category: cs
---

I assemble my steps for setting up a [Keras](https://keras.io/) environment with [tensorflow](https://www.tensorflow.org/) with one NVIDIA GPU on a vanilla Ubuntu 16.04 image. When you are interested in exploring deep neuronal networks, but you do not have a capable PC at home / work or want to scale the number of GPUs, cloud GPUs become very interesting.<!--more-->

Setting up a virtual machine on google cloud compute engine (or whichever cloud provider you favor) on the basis of *n1-standard-4 (4 vCPUs, 15 GB memory)* and customize it with one *NVIDIA Tesla K80*. I use Ubuntu 16.04 LTS image. `ssh` onto that machine and install proper python:

```sh
» sudo apt update
» sudo apt upgrade -y
» sudo apt install python3 python3-pip -y
» pip3 install --user pipenv
```

* Use Python3 instead of Python2, see [PEP 373](https://www.python.org/dev/peps/pep-0373/).
* Use [`pipenv`](https://packaging.python.org/tutorials/managing-dependencies) for your projects.

For installing tensorflow 1.5 with GPU support ([instructions](https://www.tensorflow.org/install/install_linux)), we need cuda 9.0, cuDNN 7.0 and libcupti-dev. The next steps set up the GPU and cuda environment. I follow some steps detailed [here](https://askubuntu.com/a/967333):

```sh
# install current NVIDIA drivers
» sudo add-apt-repository ppa:graphics-drivers/ppa
» sudo apt update
» sudo apt install nvidia-390 nvidia-390-dev -y
# check driver is running
» lsmod | grep nvidia
# if no output appears from lsmod, start driver by: 
» sudo modprobe nvidia
» nvidia-smi  # should show you your GPU

# prepare cuda 9 and sdk
sudo apt install freeglut3-dev build-essential \
                 libx11-dev libxmu-dev libxi-dev \
                 libglu1-mesa libglu1-mesa-dev

# download cuda toolkit
» wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
» sudo sh cuda_9.0.176_384.81_linux-run
# follow the install dialog, for one question I said no:
# Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
# (y)es/(n)o/(q)uit: n

# install patches to cuda 9.0
» wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/cuda_9.0.176.1_linux-run
» sudo sh cuda_9.0.176.1_linux-run

# install libcupti-dev (for CUDA >= 9.0)
» # sudo apt-get install cuda-command-line-tools
# should already be installed this way, just add the LD_LIBRARY_PATH, see below

# put also into .bashrc
» export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
» export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
» export PATH=/usr/local/cuda/bin/:$PATH
» export CUDA_HOME=/usr/local/cuda
# try cuda installation, e.g. via https://gist.github.com/dpiponi/1502434

# following some optimization guidelines from google:
# https://cloud.google.com/compute/docs/gpus/add-gpus#gpu-performance
» sudo nvidia-smi -pm 1
Enabled persistence mode for GPU 00000000:00:04.0.
All done.
# for Tesla K80 GPU
» sudo nvidia-smi -ac 2505,875
Applications clocks set to "(MEM 2505, SM 875)" for GPU 00000000:00:04.0
All done.

# for Tesla K100 GPU
» sudo nvidia-smi -ac 715,1328

» sudo nvidia-smi --auto-boost-default=DISABLED
All done.
```

Further we need cuDNN 7 (you are required to create a NVIDIA Developer account). Go to https://developer.nvidia.com/cudnn (create the account) and follow survey to download `Download cuDNN v7.0.5 (Dec 5, 2017), for CUDA 9.0` -> `cuDNN v7.0.5 Library for Linux`. This downloads onto your local machine - `scp` it to your server:

```sh
» scp cudnn-8.0-linux-x64-v6.0.tgz tammo.ippen@<ip-of-server>:/home/tammo.ippen
# then follow cuDNN install instructions:
» tar -xzvf cudnn-9.0-linux-x64-v7.tgz
» sudo cp cuda/include/cudnn.h /usr/local/cuda/include
» sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
» sudo chmod a+r /usr/local/cuda/include/cudnn.h \
               /usr/local/cuda/lib64/libcudnn*

```

Now we setup Keras and tensorflow. As suggested by TF install instructions, we install into a `virtualenv`, which is easy, given `pipenv`:

```sh
» mkdir ml
» cd ml
» pipenv --three install tensorflow-gpu keras ipython
# you can see, what is installed by:
» pipenv graph
ipython==6.2.1
  - decorator [required: Any, installed: 4.2.1]
  - jedi [required: >=0.10, installed: 0.11.1]
    - parso [required: ==0.1.1, installed: 0.1.1]
  - pexpect [required: Any, installed: 4.4.0]
    - ptyprocess [required: >=0.5, installed: 0.5.2]
  - pickleshare [required: Any, installed: 0.7.4]
  - prompt-toolkit [required: >=1.0.4,<2.0.0, installed: 1.0.15]
    - six [required: >=1.9.0, installed: 1.11.0]
    - wcwidth [required: Any, installed: 0.1.7]
  - pygments [required: Any, installed: 2.2.0]
  - setuptools [required: >=18.5, installed: 38.5.1]
  - simplegeneric [required: >0.8, installed: 0.8.1]
  - traitlets [required: >=4.2, installed: 4.3.2]
    - decorator [required: Any, installed: 4.2.1]
    - ipython-genutils [required: Any, installed: 0.2.0]
    - six [required: Any, installed: 1.11.0]
Keras==2.1.4
  - numpy [required: >=1.9.1, installed: 1.14.1]
  - pyyaml [required: Any, installed: 3.12]
  - scipy [required: >=0.14, installed: 1.0.0]
    - numpy [required: >=1.8.2, installed: 1.14.1]
  - six [required: >=1.9.0, installed: 1.11.0]
tensorflow-gpu==1.5.0
  - absl-py [required: >=0.1.6, installed: 0.1.10]
    - six [required: Any, installed: 1.11.0]
  - numpy [required: >=1.12.1, installed: 1.14.1]
  - protobuf [required: >=3.4.0, installed: 3.5.1]
    - setuptools [required: Any, installed: 38.5.1]
    - six [required: >=1.9, installed: 1.11.0]
  - six [required: >=1.10.0, installed: 1.11.0]
  - tensorflow-tensorboard [required: <1.6.0,>=1.5.0, installed: 1.5.1]
    - bleach [required: ==1.5.0, installed: 1.5.0]
      - html5lib [required: !=0.9999,<0.99999999,>=0.999,!=0.99999, installed: 0.9999999]
        - six [required: Any, installed: 1.11.0]
      - six [required: Any, installed: 1.11.0]
    - html5lib [required: ==0.9999999, installed: 0.9999999]
      - six [required: Any, installed: 1.11.0]
    - markdown [required: >=2.6.8, installed: 2.6.11]
    - numpy [required: >=1.12.0, installed: 1.14.1]
    - protobuf [required: >=3.4.0, installed: 3.5.1]
      - setuptools [required: Any, installed: 38.5.1]
      - six [required: >=1.9, installed: 1.11.0]
    - six [required: >=1.10.0, installed: 1.11.0]
    - werkzeug [required: >=0.11.10, installed: 0.14.1]
    - wheel [required: >=0.26, installed: 0.30.0]
  - wheel [required: >=0.26, installed: 0.30.0]
```

Verify tensorflow is working as expected:

```python
# activate the environment and start coding
» pipenv shell --fancy
# maybe you have to set CUDA_VISIBLE_DEVICES, if no Session can be created
» export CUDA_VISIBLE_DEVICES=gpu:0
» ipython
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import tensorflow as tf
   ...: hello = tf.constant('Hello, TensorFlow!')
   ...: sess = tf.Session()
   ...: print(sess.run(hello))
   ...:
2018-02-22 16:17:58.898393: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-02-22 16:17:58.901724: E tensorflow/stream_executor/cuda/cuda_driver.cc:406] failed call to cuInit: CUDA_ERROR_NO_DEVICE
2018-02-22 16:17:58.901773: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: dev-tammo
2018-02-22 16:17:58.901788: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: dev-tammo
2018-02-22 16:17:58.901826: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: 390.25.0
2018-02-22 16:17:58.901930: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:369] driver version file contents: """NVRM version: NVIDIA UNIX x86_64 Kernel Module  390.25  Wed Jan 24 20:02:43 PST 2018
GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.9)
"""
2018-02-22 16:17:58.901957: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: 390.25.0
2018-02-22 16:17:58.901970: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:300] kernel version seems to match DSO: 390.25.0
b'Hello, TensorFlow!'

In [5]: import keras
Using TensorFlow backend.
```

Happy coding. :tada:
