#!/bin/bash

workers=8
saved_folder='./comp_hp_search'

name="${1}_${2}"
d=$3
b=$4
lr=$5
wd=$6
ep=$7
b1=$8
b2=$9
m=${10}
dr=${11}

echo "Starting $name with $workers workers"

python3 train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --batch_size $b \
    --lr $lr \
    --lr0 $lr \
    --weight_decay $wd \
    --epoch_decay $ep \
    --margin $m \
    --loss comp \
    --betas $b1 $b2 \
    --augmentations "convirt"\
    --aug_args "h.ra.et"\
    --dropout $dr 
    # --early_stopping_patience 2
