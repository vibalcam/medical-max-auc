#!/bin/bash

workers=14
saved_folder='./other'

d=pneumoniamnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

name=$d

# python train.py \
#     --name default \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 100 \
#     --lr_steps 50 75 \
#     --batch_size 128 \
#     --lr 1e-3 \
#     --loss_type bce \
#     --augmentations basic \
#     --use_best_model

## best val auc
python train.py \
    --name auc_val \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 1e-3 \
    --epoch_decay 0.03 \
    --margin 1.0 \
    --loss auc \
    --augmentations basic \
    --aug_args '' \
    --dropout 0 \
    --resize 128
