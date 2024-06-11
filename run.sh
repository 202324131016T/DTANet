#!/bin/bash

#cd /data/wyl/DTANet/
#conda activate DTANet

datatime=$(date +"%Y%m%d-%H%M%S")

CUDA_VISIBLE_DEVICES=0 \
python run_experiments.py \
--log_dir ./logs/${datatime}/ \
--checkpoint_path ./checkpoints/${datatime}/ \
--result ./result/${datatime}/ \
> ./logs/train_${datatime}_run_experiments.txt # log path

#datatime=20231230-144959
## only test
#CUDA_VISIBLE_DEVICES=0 \
#python run_experiments.py \
#--log_dir ./logs/"${datatime}_test/" \
#--checkpoint_path ./checkpoints/${datatime}/ \
#--result ./result/"${datatime}_test/" \
#--only_test True \
#> ./logs/"test_${datatime}_run_experiments.txt" # log path
