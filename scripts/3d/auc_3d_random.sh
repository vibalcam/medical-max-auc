#!/bin/bash

workers=7
saved_folder='./saved_models'
name=basic_hp

d=$1

python run_3d.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --loss auc \
    --augmentations basic \
    --aug_args '' \
    --type_3d '3d'
