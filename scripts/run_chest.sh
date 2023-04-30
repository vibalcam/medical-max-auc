#!/bin/bash

workers=14
saved_folder='./saved_models'

d=chestmnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]


name=chestmnist
python train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --batch_size 256 \
    --lr 1e-3 \
    --loss bce \
    --augmentations convirt \
    --aug_args ra.gb \
    --dropout 0

# python train.py \
#         --name basic \
#         --dataset $d \
#         --save_dir $saved_folder \
#         --workers $workers \
#         --seed 123456 \
#         --epochs 100 \
#         --lr_steps 50 75 \
#         --batch_size 128 \
#         --lr 1e-3 \
#         --loss bce \
#         --augmentations basic