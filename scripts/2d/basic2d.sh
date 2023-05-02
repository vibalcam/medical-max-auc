#!/bin/bash

workers=14

saved_folder='./logs'
log_folder='./logs_txt'

datasets=("breastmnist" "pneumoniamnist" "chestmnist")
#  "nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

# loop through indices of datasets
for (( i=0; i<${#datasets[@]}; i++ ))
do

    d="${datasets[$i]}"

    # basic training
    echo "Starting basic training for dataset $d"
    python train.py \
        --name default \
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
        --use_best_model
done
