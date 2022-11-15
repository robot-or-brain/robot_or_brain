# robot_or_brain

## Installation of the Jupyter notebook requirements

### Add shared group and assign as primary to users, change permissions on data folder to group
```
sudo groupadd robot
sudo usermod -g robot cmeijer
sudo usermod -g robot mvanmeersb
sudo chown -R cmeijer:robot /data/volume_2/robot_or_brain/
sudo chmod -R g+rwx /data/volume_2/robot_or_brain/
```

### Install Miniconda
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh
conda create --name tf python=3.9 
```
Close and reopen the terminal window (relogin) in order for the installation to have full effect (e.g. setting PATH variable).

**Install ipykernel _inside_ the conda environment**
```
conda install --yes --channel anaconda ipykernel
python3 -m ipykernel install --user --name tf --display-name "Tensorflow"
vim ~/.local/share/jupyter/kernels/tf/kernel.json
```
**Change the location of your python installation so that the file reads:**
```
{
 "argv": [
  "~/miniconda3/envs/tf/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Tensorflow",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```

*For all of the below, _make sure_ we are in the right conda environment*
```
conda activate tf 
```
*Your prompt should now look something like this ... note the (tf) in front of it:*
```
(tf) mvanmeersb@robot:/data/volume_2/robot_or_brain$ 
```

### Install Tensorflow
**1. Verify that the Nvidia System Management Interface is installed**
```
nvidia-smi
```
**The output should look like this:**
```
Wed Jun 15 09:01:40 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.32.00    Driver Version: 455.32.00    CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  On   | 00000000:00:05.0 Off |                  N/A |
| 30%   29C    P8     6W / 250W |      1MiB / 11019MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

**2. Install the CUDA Deep Neural Network library**
```
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ 
pip install --upgrade pip 
pip install tensorflow 
```
**update /etc/jupyterhub/jupyterhub_config.py to accept LD_LIBRARY_PATH**
```
import os
c.Spawner.notebook_dir = '~'
c.Spawner.default_url = '/lab?reset'
c.Authenticator.admin_users = {'ubuntu'}

os.environ['LD_LIBRARY_PATH'] = '~/miniconda3/envs/tf/lib/'
c.Spawner.env.update('LD_LIBRARY_PATH')
c.Spawner.env_keep.append('LD_LIBRARY_PATH')
```

**3. Verify the Tensorflow install on the CPU**
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))" 
```
**The output should look like this:**
```
2022-06-15 09:05:47.364161: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2022-06-15 09:05:48.705821: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    <...>
2022-06-15 09:05:49.571063: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9648 MB memory:  -> device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:00:05.0, compute capability: 7.5
tf.Tensor(-1127.574, shape=(), dtype=float32)
```

**4. Verify access of Tensorflow to the GPU**
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 
```
**The output should look like this:**
```
2022-06-15 09:10:28.374687: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2022-06-15 09:10:29.531149: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    <...>
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Install Weights and Biases
``` 
pip install wandb 
export WANDB_PATH=~/miniconda3/envs/tf/bin/wandb
```
**Login to weights and biases to create your profile locally**
```
wandb login
```
Open a website to https://wandb.ai/home  
Go to your profile -> Settings  
Scroll down to API keys and create ... or copy the API key to your terminal  
