#!/bin/bash

workers=8

saved_folder='./logs'
log_folder='./logs_txt'

datasets=("breastmnist" "pneumoniamnist")
#  "nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

# loop through indices of datasets
for (( i=0; i<${#datasets[@]}; i++ ))
do

    d="${datasets[$i]}"

    # basic training
    echo "Starting basic training for dataset $d"
    python3 train_ensemble2.py \
        --name default \
        --dataset $d \
        --save_dir $saved_folder \
        --workers $workers \
        --seed 123456 \
        --epochs 10 \
        --lr_steps 50 75 \
        --batch_size 128 \
        --lr 1e-3 \
        --loss auc \
        --augmentations basic \
        --num_models 3
done
