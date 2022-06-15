# robot_or_brain

## Installation of the Jupyter notebook requirements

### Install Miniconda
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh 
```

### Install Tensorflow in a new conda environment
```
conda create --name tf python=3.9 
<restart the terminal so conda will be in the search path> 
    1  conda 
    2  conda create --name tf python=3.9 
    3  conda activate tf 
    4  nvidia-smi 
    5  conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 
    6  which pip 
    7  pip install wandb 
    8  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ 
    9  export WANDB_PATH=$/home/cmeijer/.local/bin/wandb 
   10  pip install --upgrade pip 
   11  pip install tensorflow 
   12  python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))" 
   13  python3 -c "import tensorflow as tf; print('printing ', tf.reduce_sum(tf.random.normal([1000, 1000])))" 
   14  python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 
```
