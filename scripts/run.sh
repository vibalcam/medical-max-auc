#!/bin/bash

workers=16
saved_folder='./saved_models'
name=basic

d=breastmnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

## pauc
# lr {1e-3, 1e-4, 1e-5}, 5e-6
# wd 2e-4, 1e-2
# gammas {0.9, 0.5}
# lambda, tau {0.1, 1, 10}
python train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 100 \
    --lr_steps 50 75 \
    --batch_size 128 \
    --lr 1e-3 \
    --loss bce \
    --augmentations basic
