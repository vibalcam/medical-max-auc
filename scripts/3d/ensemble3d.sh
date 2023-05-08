#!/bin/bash

workers=3

saved_folder='./ensemble_train_and_val'
log_folder='./logs_txt'

datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

# loop through indices of datasets
for (( i=0; i<${#datasets[@]}; i++ ))
do

    d="${datasets[$i]}"

    # basic training
    echo "Starting basic training for dataset $d"
    python3 train_ensemble.py \
        --name default \
        --dataset $d \
        --save_dir $saved_folder \
        --workers $workers \
        --seed 123456 \
        --epochs 50 \
        --lr_steps 50 75 \
        --batch_size 8 \
        --lr 1e-1 \
        --weight_decay 1e-4 \
        --epoch_decay 3e-2 \
        --margin 0.6 \
        --loss auc \
        --augmentations basic \
        --type_3d '3d' \
        --use_16 \
        --train_on_val 'true' \
        --early_stopping_patience 5
done
