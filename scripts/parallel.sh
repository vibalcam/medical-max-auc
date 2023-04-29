#!/bin/bash

name=basic
workers=16

saved_folder='./saved_models'
log_folder='./logs'

datasets=("breastmnist" "pneumoniamnist" "chestmnist" "nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

# loop through indices of datasets
for (( i=0; i<${#datasets[@]}; i++ ))
do
    # launch up to 10 processes in parallel
    while (( $(pgrep -c -f train.py) >= 10 )); do sleep 60; done
    d="${datasets[$i]}"

    echo "Starting training for dataset $d"
    python train.py \
        --name $name \
        --dataset $d \
        --save_dir $saved_folder \
        --workers $workers \
        --seed 123456 \
        --epochs 100 \
        --lr_steps 50 75 \
        --batch_size 128 \
        --lr 1e-3 \
        --loss bce \
        --augmentations basic \
        > "${log_folder}/${name}/${d}.log" &
done
