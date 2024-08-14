# DTANet

> **Paper: Dual Interaction and Kernel-Diverse Network for Accurate Drug-Target Binding Affinity Prediction**
> 
> Author: Yulong Wu, Jin Xie, Jing Nie, Jian Hu and Yuansong Zeng

## Installation

```
conda create -n DTANet python=3.8
conda activate DTANet
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Other requirements:
- Linux
- NVIDIA GPU
- The detailed environment configuration is shown in **environment.yml**.

## Train and Test

- The **arguments.py** provides some model parameters.
- The **datahelper.py** provides some methods for data preprocessing.
- The **emetrics.py** provides evaluation metrics for the model.
- The **figures.py** provides some visualization methods.
- The **model/model_block.py** provides the basic blocks that **model/model_att3.py** needs to use.
- The **model/model_att3.py** is the main model of our DTANet, and other **model/model\*_\*.py** are the model needed for ablation experiments.
- We use them in **run_experiments.py**.

```
python run_experiments.py
# or
bash run.sh
```


