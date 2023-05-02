#!/bin/bash

workers=6
saved_folder='./saved_models'

datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

d="${datasets[$1]}"
name=$d

## bce
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

######################################################

## auc
python train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 100 \
    --lr_steps 50 75 \
    --batch_size 32 \
    --lr $lr \
    --weight_decay $wd \
    --epoch_decay $ep \
    --margin $m \
    --loss auc \
    --augmentations basic \
    --dropout $drop \
    --type_3d '3d'
