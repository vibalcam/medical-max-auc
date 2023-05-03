#!/bin/bash

workers=7
saved_folder='./saved_local'
name=basic_hp

python run_2d.py \
    --name $name \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --loss auc \
    --augmentations basic \
    --aug_args ''
