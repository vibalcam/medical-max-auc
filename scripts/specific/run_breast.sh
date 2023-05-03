#!/bin/bash

workers=14
saved_folder='./other'

d=breastmnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

name=$d

## basic
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
# python train.py \
#     --name auc_val \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 128 \
#     --lr 0.1 \
#     --weight_decay 1e-5 \
#     --epoch_decay 0.03 \
#     --margin 1.0 \
#     --loss auc \
#     --augmentations basic \
#     --aug_args '' \
#     --dropout 0











##################################################
##################################################


# ## best test auc
# python train.py \
#     --name auc_test \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 128 \
#     --lr 0.1 \
#     --weight_decay 1e-4 \
#     --epoch_decay 0.03 \
#     --margin 0.6 \
#     --loss auc \
#     --augmentations basic \
#     --aug_args '' \
#     --dropout 0.1


# # bce
# python train.py \
#     --name "${name}" \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 256 \
#     --lr 1e-3 \
#     --loss_type bce \
#     --augmentations convirt \
#     --aug_args ra.gb \
#     --dropout 0

# # bce + auc
# python train.py \
#     --name "${name}_pre" \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 256 \
#     --lr 1e-4 \
#     --lr0 0.02 \
#     --betas 0.9 0.9 \
#     --weight_decay 1e-6 \
#     --epoch_decay 3e-5 \
#     --margin 1.0 \
#     --loss_type auc \
#     --augmentations convirt \
#     --aug_args ra \
#     --dropout 0 \
#     --pretrained 'saved_models/breastmnist/breastmnist/convirt_bce_128/version_0/last.ckpt'

# # bce + auc
# python train.py \
#     --name "${name}_pre" \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 256 \
#     --lr 0.01 \
#     --lr0 0.02 \
#     --betas 0.9 0.9 \
#     --weight_decay 1e-6 \
#     --epoch_decay 3e-5 \
#     --margin 1.0 \
#     --loss_type comp \
#     --augmentations convirt \
#     --aug_args ra \
#     --dropout 0 \
#     --pretrained 'saved_models/breastmnist/breastmnist/convirt_bce_128/version_0/last.ckpt'

######################################

# ## auc
## python train.py \
##     --name "${name}" \
##     --dataset $d \
##     --save_dir $saved_folder \
##     --workers $workers \
##     --seed 123456 \
##     --epochs 200 \
##     --lr_steps 100 150 \
##     --batch_size 128 \
##     --lr 1e-2 \
##     --weight_decay 1e-4 \
##     --epoch_decay 3e-3 \
##     --margin 1.0 \
##     --loss_type auc \
##     --augmentations basic \
##     --aug_args ra.gb \
##     --dropout 0

## auc
# python train.py \
#     --name "${name}" \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 256 \
#     --lr 1e-2 \
#     --weight_decay 1e-4 \
#     --epoch_decay 3e-2 \
#     --margin 1.0 \
#     --loss_type comp \
#     --augmentations convirt \
#     --aug_args ra \
#     --dropout 0

# python train.py \
#     --name auc_val \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 128 \
#     --lr 0.1 \
#     --lr0 0.2 \
#     --betas 0.9 0.9 \
#     --weight_decay 1e-5 \
#     --epoch_decay 0.03 \
#     --margin 1.0 \
#     --loss comp \
#     --augmentations basic \
#     --aug_args '' \
#     --dropout 0

######################################

# python train.py \
#     --name "${name}" \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 125 \
#     --lr_steps 75 100 \
#     --batch_size 1024 \
#     --lr 1e-4 \
#     --loss_type pre \
#     --augmentations convirt \
#     --aug_args ra.gb.et.h \
#     --dropout 0
#     # --aug_args ra.gb.cj.et \

# python train.py \
#     --name "${name}" \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 25 150 \
#     --batch_size 256 \
#     --lr 1e-3 \
#     --loss_type bce \
#     --augmentations convirt \
#     --aug_args ra.gb \
#     --dropout 0 \
#     --pretrained 'saved_models/breastmnist/breastmnist_auc/convirt/pre/version_0/last.ckpt' \
#     --warmup_epochs 25

##########################################
##########################################

## auc
# python train.py \
#     --name "${name}" \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 256 \
#     --lr 1e-2 \
#     --weight_decay 1e-4 \
#     --epoch_decay 3e-2 \
#     --margin 1.0 \
#     --loss_type comp \
#     --augmentations convirt \
#     --aug_args ra \
#     --dropout 0
# lr=0.01, 
# lr0=0.02,
# beta1=0.9,
# beta2=0.9,
