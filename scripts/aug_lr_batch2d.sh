#!/bin/bash

workers=8
saved_folder='./saved_models'

# d=breastmnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

datasets=("breastmnist" "pneumoniamnist" "chestmnist")
lrs=(1e-2 1e-3 1e-4)
bath_sizes=(128 256 512)
wds=(1e-2 1e-4)
augs=(ra.gb ra.cj ra.et ra.gb.cj)
name=lr_aug_batch

for d in "${datasets[@]}";do
    for lr in "${lrs[@]}";do
        for wd in "${wds[@]}";do
            for aug in "${augs[@]}";do
                for b in "${bath_sizes[@]}";do
                    # basic training
                    python train.py \
                    --name $name \
                    --dataset $d \
                    --save_dir $saved_folder \
                    --workers $workers \
                    --seed 123456 \
                    --epochs 200 \
                    --lr_steps 100 150 \
                    --batch_size $b \
                    --lr $lr \
                    --weight_decay $wd \
                    --loss bce \
                    --augmentations convirt \
                    --aug_args $aug \
                    --dropout 0 \
                    > "logs_txt/${name}_${d}_${lr}_${wd}_${aug}_${b}.txt"
                done
            done
        done
    done
done
