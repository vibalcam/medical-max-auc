#!/bin/bash

workers=8
# saved_folder='./other'
saved_folder='./train_and_val'

d=chestmnist
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

# best val auc
python train.py \
	--name auc_val \
	--dataset $d \
	--save_dir $saved_folder \
	--workers $workers \
	--seed 123456 \
	--epochs 200 \
	--lr_steps 100 150 \
	--batch_size 128 \
	--lr 1e-2 \
	--weight_decay 1e-5 \
	--epoch_decay 0.03 \
	--margin 1.0 \
	--loss auc \
	--augmentations convirt \
    --aug_args 'cj' \
	--dropout 0 \
	--evaluate_every 5 \
	--early_stopping_patience 10 \
	--use_16 \
	--use_best_model

# # train on validation and train datasets
# python3 train.py \
# 	--name auc_val \
# 	--dataset $d \
# 	--save_dir $saved_folder \
# 	--workers $workers \
# 	--seed 123456 \
# 	--epochs 200 \
# 	--lr_steps 100 150 \
# 	--batch_size 128 \
# 	--lr 1e-2 \
# 	--weight_decay 1e-5 \
# 	--epoch_decay 0.03 \
# 	--margin 1.0 \
# 	--loss auc \
# 	--augmentations convirt \
#     --aug_args 'cj' \
# 	--dropout 0 \
# 	--evaluate_every 5 \
# 	--early_stopping_patience 5 \
# 	--use_16 \
# 	--train_on_val 'true' \
# 	--use_best_model