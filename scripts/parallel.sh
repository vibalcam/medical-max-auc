#!/bin/bash

name=basic
workers=14

saved_folder='./logs'
log_folder='./logs_txt'

datasets=("breastmnist" "pneumoniamnist" "chestmnist" "nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

# exec 9<> lock_file.lock
# flock -n 9 || exit 1

# loop through indices of datasets
for (( i=0; i<${#datasets[@]}; i++ ))
do
    # acquire a lock on every third iteration
    # if (( i % 4 == 0 )); then
    #     exec 8<>"train_file.lock"
    #     flock -x 8
    # fi

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
        > "${log_folder}/${name}_${d}.log" &

    sleep 3 &

    # release the lock on the train file
    if (( (i + 1) % 3 == 0 )); then
        wait
        # flock -u 8
    fi
done

# # close the lock file
# flock -u 9
# exec 9>&-
