#!/bin/bash

workers=6
saved_folder='./saved_models'

d=synapsemnist3d
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

# try 1e-3
name=3d
python train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 150 \
    --lr_steps 50 75 125 \
    --batch_size 64 \
    --lr 1e-2 \
    --loss bce \
    --augmentations basic \
    --aug_args ra.gb \
    --dropout 0.1


# python train.py \
#     --name $name \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 150 \
#     --lr_steps 50 75 125 \
#     --batch_size 64 \
#     --lr 1e-2 \
#     --loss bce \
#     --augmentations basic \
#     --aug_args ra.gb \
#     --dropout 0 \
#     --type_3d channels
