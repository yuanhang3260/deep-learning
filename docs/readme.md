## aliyun gpu ecs

- console

```
https://ecs.console.aliyun.com/server/region/cn-hangzhou
```

- install nvidia driver

```
https://help.aliyun.com/zh/egs/user-guide/install-a-gpu-driver-on-a-gpu-accelerated-compute-optimized-linux-instance?spm=a2c4g.11186623.0.0.414e2ef7BDrp42

# for shared-gpu ecs
https://help.aliyun.com/zh/egs/user-guide/use-cloud-assistant-to-automatically-install-and-upgrade-grid-drivers
```

- mount new disk

```
https://help.aliyun.com/zh/ecs/user-guide/attach-a-data-disk#d903bdbaaez3q
```

```bash
# mount
sudo mount /dev/vdb1 ~/data1
# change owner
sudo chown -R ecs-user:ecs-user ~/data1
```

- g++ 8

```bash
scl enable devtoolset-8 -- bash
```

- upgrade glibc

```
https://github.com/apernet/tcp-brutal/issues/7
```

- bashrc

```bash
export JAVA_HOME="/usr/share/jdk"
export CUDA_HOME="/usr/local/cuda"
export PATH="$JAVA_HOME/bin:$CUDA_HOME/bin:$HOME/data1/miniconda3/bin:$PATH"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```



## conda

- linux install conda

```
https://docs.anaconda.com/free/miniconda/#quick-command-line-install
```

## jupyter

- install jupyter

```bash
conda install jupyter
#conda install -c conda-forge jupyterlab
```

- setup jupyter server

```bash
https://developer.aliyun.com/article/1436710

nohup jupyter notebook --allow-root > jupyter.log 2>&1 &
nohup jupyter lab --allow-root > jupyter.log 2>&1 &
```

- juputer lab web

```bash
http://${ip}:8888/lab/
```



## tensorflow

- install tensorflow-gpu

```bash
CONDA_OVERRIDE_CUDA="11.2" (on cpu-only machine)

conda install -c conda-forge tensorflow-gpu=2.16.0
```

- accelerate tensorflow gpu startup (optional)

```bash
export CUDA_CACHE_MAXSIZE=4294967296
```

- set device

```python
def set_device(device):
    if device == 'cpu':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == 'gpu':
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

set_device('cpu')
```

