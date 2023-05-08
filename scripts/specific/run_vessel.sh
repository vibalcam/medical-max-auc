#!/bin/bash

workers=14
saved_folder='./other'

d=vesselmnist3d
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

name=$d

# python train.py \
#     --name basic \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 100 \
#     --lr_steps 50 75 \
#     --batch_size 32 \
#     --lr 1e-3 \
#     --loss bce \
#     --augmentations basic \
#     --use_best_model \
#     --type_3d '3d'

# ## best val auc
python train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --batch_size 32 \
    --lr 0.1 \
    --weight_decay 1e-5 \
    --epoch_decay 3e-3 \
    --margin 1.0 \
    --loss auc \
    --augmentations convirt \
    --aug_args 'rc' \
    --dropout 0 \
    --type_3d '3d' \
    --evaluate_every 5 \
    --early_stopping_patience 10 \
    --resize 80 \
    --use_16

###################################################
###################################################

# ## best val auc with dropout
# python train.py \
#     --name $name \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 32 \
#     --lr 0.1 \
#     --weight_decay 1e-5 \
#     --epoch_decay 1e-3 \
#     --margin 1.0 \
#     --loss auc \
#     --augmentations convirt \
#     --aug_args $1 \
#     --dropout 0.1 \
#     --type_3d '3d' \
#     --evaluate_every 5 \
#     --early_stopping_patience 10
