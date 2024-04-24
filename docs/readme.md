## aliyun gpu ecs

- console

```
https://ecs.console.aliyun.com/server/region/cn-hangzhou
```

- install nvidia driver

```
https://help.aliyun.com/zh/egs/user-guide/use-cloud-assistant-to-automatically-install-and-upgrade-grid-drivers
```

## conda

- linux install conda

```
https://docs.anaconda.com/free/miniconda/#quick-command-line-install
```

## Jupyter

- install jupyter

```bash
conda install jupyter
conda install -c conda-forge jupyterlab
```

- setup jupyter server

```bash
https://developer.aliyun.com/article/1436710

nohup jupyter notebook --allow-root > jupyter.log 2>&1 &
nohup jupyter lab --allow-root > jupyter.log 2>&1 &
```

## tensorflow

- install tensorflow-gpu

```bash
CONDA_OVERRIDE_CUDA="11.2" (on cpu-only machine)

conda install -c conda-forge tensorflow-gpu=2.10.0
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

