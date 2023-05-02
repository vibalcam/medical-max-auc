#!/bin/bash

workers=14
saved_folder='./saved_models'

d=chestmnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]


name=chestmnist

python train.py \
        --name "${name}" \
        --dataset $d \
        --save_dir $saved_folder \
        --workers $workers \
        --seed 123456 \
        --epochs 20 \
        --lr_steps 10 \
        --batch_size 2048 \
        --lr 1e-4 \
        --loss_type pre \
        --augmentations convirt \
        --aug_args rc.ra.et.cj \
        --sampler

python train.py \
        --name default \
        --dataset $d \
        --save_dir $saved_folder \
        --workers $workers \
        --seed 123456 \
        --epochs 100 \
        --lr_steps 50 75 \
        --batch_size 256 \
        --lr 1e-3 \
        --loss bce \
        --augmentations basic \
        --use_best_model


## basic
# python train.py \
#         --name basic \
#         --dataset $d \
#         --save_dir $saved_folder \
#         --workers $workers \
#         --seed 123456 \
#         --epochs 20 \
#         --batch_size 512 \
#         --lr 1e-4 \
#         --loss_type bce \
#         --augmentations convirt \
#         --aug_args ra.et.gb.cj \
#         --use_best_model \
#         --evaluate_every 1
#         # --dropout 0.3 \
#         # --pretrained 'saved_models/chestmnist/chestmnist/convirt_pre_0.0001_0.01_100_2048/version_0/last.ckpt'


# python train.py \
#         --name basic \
#         --dataset $d \
#         --save_dir $saved_folder \
#         --workers $workers \
#         --seed 123456 \
#         --epochs 50 \
#         --lr_steps 50 \
#         --batch_size 512 \
#         --lr 1e-2 \
#         --margin 0.7 \
#         --epoch_decay 3e-3 \
#         --weight_decay 1e-4 \
#         --loss_type auc \
#         --augmentations convirt \
#         --aug_args ra.et.gb.cj \
#         --use_best_model \
#         --evaluate_every 1
        # --pretrained 'saved_models/chestmnist/chestmnist/convirt_pre_0.0001_0.01_125_2048/version_0/last.ckpt'
