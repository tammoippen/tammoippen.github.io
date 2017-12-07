---
layout: post
title: Setting up a Keras / tensorflow environment in a vanilla Ubuntu
last_modified_at: 2017-12-07T12:00:00+01:00
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

For installing tensorflow with GPU support ([instructions](https://www.tensorflow.org/install/install_linux)), we need cuda 8.0 (not 9.0), cuDNN 6.0 (not 7.0) and libcupti-dev. The next steps set up the GPU and cuda environment. I follow some steps detailed [here](https://askubuntu.com/a/967333):

```sh
# install current NVIDIA drivers
» sudo add-apt-repository ppa:graphics-drivers/ppa
» sudo apt update
» sudo apt install nvidia-387 nvidia-387-dev -y
# check driver is running
» lsmod | grep nvidia
# if no output appears from lsmod, start driver by: 
» sudo modprobe nvidia
» nvidia-smi  # should show you your GPU

# prepare cuda 8 and sdk
sudo apt install freeglut3-dev build-essential \
                 libx11-dev libxmu-dev libxi-dev \
                 libglu1-mesa libglu1-mesa-dev

# download cuda toolkit
» wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
» sudo sh cuda_8.0.61_375.26_linux.run
# follow the install dialog, for one question I said no:
# Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
# (y)es/(n)o/(q)uit: n

# install patches to cuda 8.0
» wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run
» sudo sh cuda_8.0.61.2_linux.run

# install libcupti-dev
» sudo apt install libcupti-dev

export PATH=/usr/local/cuda/bin/:$PATH  # put also into .bashrc
# try cuda installation, e.g. via https://gist.github.com/dpiponi/1502434

# following some optimization guidelines from google (Tesla K80):
# https://cloud.google.com/compute/docs/gpus/add-gpus#gpu-performance
» sudo nvidia-smi -pm 1
Enabled persistence mode for GPU 00000000:00:04.0.
All done.
» sudo nvidia-smi -ac 2505,875
Applications clocks set to "(MEM 2505, SM 875)" for GPU 00000000:00:04.0
All done.
» sudo nvidia-smi --auto-boost-default=DISABLED
All done.
```

Further we need cuDNN 6 (you are required to create a NVIDIA Developer account). Go to https://developer.nvidia.com/cudnn (create the account) and follow survey to download `Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0` -> `cuDNN v6.0 Library for Linux`. This downloads onto your local machine - `scp` it to your server:

```sh
» scp cudnn-8.0-linux-x64-v6.0.tgz tammo.ippen@<ip-of-server>:/home/tammo.ippen
# then follow cuDNN install instructions:
» tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
» sudo cp cuda/include/cudnn.h /usr/local/cuda/include
» sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
» sudo chmod a+r /usr/local/cuda/include/cudnn.h \
               /usr/local/cuda/lib64/libcudnn*
# These environment variables are required according to TF install:
» export CUDA_HOME=/usr/local/cuda  # put also into .bashrc
» export LD_LIBRARY_PATH=/usr/local/cuda/lib64  # put also into .bashrc
```

Now we setup Keras and tensorflow. As suggested by TF install instructions, we install into a `virtualenv`, which is easy, given `pipenv`:

```sh
» mkdir ml
» cd ml
» pipenv --three install tensorflow-gpu keras ipython
# you can see, what is installed by:
» pipenv graph
ipython==6.2.1
  - decorator [required: Any, installed: 4.1.2]
  - jedi [required: >=0.10, installed: 0.11.0]
    - parso [required: ==0.1.0, installed: 0.1.0]
  - pexpect [required: Any, installed: 4.3.0]
    - ptyprocess [required: >=0.5, installed: 0.5.2]
  - pickleshare [required: Any, installed: 0.7.4]
  - prompt-toolkit [required: >=1.0.4,<2.0.0, installed: 1.0.15]
    - six [required: >=1.9.0, installed: 1.11.0]
    - wcwidth [required: Any, installed: 0.1.7]
  - pygments [required: Any, installed: 2.2.0]
  - setuptools [required: >=18.5, installed: 38.2.4]
  - simplegeneric [required: >0.8, installed: 0.8.1]
  - traitlets [required: >=4.2, installed: 4.3.2]
    - decorator [required: Any, installed: 4.1.2]
    - ipython-genutils [required: Any, installed: 0.2.0]
    - six [required: Any, installed: 1.11.0]
Keras==2.1.2
  - numpy [required: >=1.9.1, installed: 1.13.3]
  - pyyaml [required: Any, installed: 3.12]
  - scipy [required: >=0.14, installed: 1.0.0]
    - numpy [required: >=1.8.2, installed: 1.13.3]
  - six [required: >=1.9.0, installed: 1.11.0]
tensorflow-gpu==1.4.0
  - enum34 [required: >=1.1.6, installed: 1.1.6]
  - numpy [required: >=1.12.1, installed: 1.13.3]
  - protobuf [required: >=3.3.0, installed: 3.5.0.post1]
    - setuptools [required: Any, installed: 38.2.4]
    - six [required: >=1.9, installed: 1.11.0]
  - six [required: >=1.10.0, installed: 1.11.0]
  - tensorflow-tensorboard [required: >=0.4.0rc1,<0.5.0, installed: 0.4.0rc3]
    - bleach [required: ==1.5.0, installed: 1.5.0]
      - html5lib [required: !=0.99999,>=0.999,!=0.9999,<0.99999999, installed: 0.9999999]
        - six [required: Any, installed: 1.11.0]
      - six [required: Any, installed: 1.11.0]
    - html5lib [required: ==0.9999999, installed: 0.9999999]
      - six [required: Any, installed: 1.11.0]
    - markdown [required: >=2.6.8, installed: 2.6.9]
    - numpy [required: >=1.12.0, installed: 1.13.3]
    - protobuf [required: >=3.3.0, installed: 3.5.0.post1]
      - setuptools [required: Any, installed: 38.2.4]
      - six [required: >=1.9, installed: 1.11.0]
    - six [required: >=1.10.0, installed: 1.11.0]
    - werkzeug [required: >=0.11.10, installed: 0.12.2]
    - wheel [required: >=0.26, installed: 0.30.0]
  - wheel [required: >=0.26, installed: 0.30.0]
```

Verify tensorflow is working as expected:

```python
# activate the environment and start coding
» pipenv shell --fancy
» ipython
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import tensorflow as tf
In [2]: hello = tf.constant('Hello, TensorFlow!')
In [3]: sess = tf.Session()
2017-12-07 11:41:07.943761: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2017-12-07 11:41:10.727162: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8755
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.10GiB
2017-12-07 11:41:10.727193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)

In [4]: print(sess.run(hello))
b'Hello, TensorFlow!'

In [5]: import keras
Using TensorFlow backend.
```

Happy coding. :tada:
