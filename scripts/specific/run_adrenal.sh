#!/bin/bash

workers=8
# saved_folder='./other'
saved_folder='./train_and_val'


d=adrenalmnist3d
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

<<<<<<< HEAD
<<<<<<< HEAD
## best val auc
=======
python train.py \
    --name pre \
    --dataset $d \
    --save_dir pre \
    --workers $workers \
    --seed 123456 \
    --epochs 50 \
    --lr_steps 25 40 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --epoch_decay 3e-3 \
    --margin 0.6 \
    --loss pre \
    --augmentations basic \
    --aug_args '' \
    --dropout 0 \
    --type_3d '3d' \
    --resize 80 \
    --use_16

python train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 100 \
    --lr_steps 50 75 \
    --batch_size 32 \
    --lr 0.1 \
    --weight_decay 1e-4 \
    --epoch_decay 3e-3 \
    --margin 0.6 \
    --loss auc \
    --augmentations basic \
    --aug_args '' \
    --dropout 0 \
    --type_3d '3d' \
    --evaluate_every 5 \
    --early_stopping_patience 10 \
    --resize 80 \
    --use_16 \
    --pretrained pre/adrenalmnist3d/pre/version_0/last.ckpt

###################################################
###################################################

# ## best val auc resized to 80
>>>>>>> augmentations
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
#     --weight_decay 1e-4 \
<<<<<<< HEAD
#     --epoch_decay 3e-2 \
#     --margin 0.6 \
#     --loss auc \
#     --augmentations basic \
#     --aug_args '' \
#     --dropout 0 \
#     --type_3d '3d' \
#     --evaluate_every 5 \
#     --early_stopping_patience 10

# train on validation and train datasets
python3 train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --batch_size 32 \
    --lr 0.1 \
    --weight_decay 1e-4 \
    --epoch_decay 3e-2 \
    --margin 0.6 \
    --loss auc \
    --augmentations convirt \
    --aug_args 'et' \
    --dropout 0 \
    --type_3d '3d' \
    --evaluate_every 5 \
    --train_on_val 'true' \
    --early_stopping_patience 5


# ## best val auc resized to 80
=======
=======
#     --epoch_decay 3e-3 \
#     --margin 0.6 \
#     --loss auc \
#     --augmentations convirt \
#     --aug_args $1 \
#     --dropout 0 \
#     --type_3d '3d' \
#     --evaluate_every 5 \
#     --early_stopping_patience 10 \
#     --resize 80 \
#     --use_16

>>>>>>> augmentations
# ## best val auc
>>>>>>> eb8ee3bcb71dbbf5f09ecc36a4fdee6e9a9de292
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
#     --weight_decay 1e-4 \
#     --epoch_decay 3e-2 \
#     --margin 0.6 \
#     --loss auc \
#     --augmentations convirt \
#     --aug_args $1 \
#     --dropout 0 \
#     --type_3d '3d' \
#     --evaluate_every 5 \
#     --early_stopping_patience 10
