#!/bin/bash

workers=14
saved_folder='./saved_models'


name=lr_aug_batch

python run3d.py \
    --name $name \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 100 \
    --lr_steps 50 75 \
    --loss auc \
    --augmentations basic \
    --type_3d '3d' \
    --use_best_model
