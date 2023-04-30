#!/bin/bash

workers=14
saved_folder='./saved_models'

d=breastmnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]


name=breastmnist
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
#     --name $name \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 256 \
#     --lr 1e-3 \
#     --loss bce \
#     --augmentations convirt \
#     --aug_args ra.gb \
#     --dropout 0
