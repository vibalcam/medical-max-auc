#!/bin/bash

workers=4
saved_folder='./saved_models'

# d=breastmnist
# ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

datasets=("breastmnist" "pneumoniamnist" "chestmnist")
lrs=(1e-2 1e-3)
bath_sizes=(256 512)
wds=(1e-2 1e-4)
eps=(3e-2 3e-3)
margin=(1.0 0.7)
dropouts=(0 0.05 0.1)
# augs=(ra ra.gb ra.et ra.gb.cj ra.et.cj)
name=lr_aug_batch

# Set the maximum number of scripts to run in parallel
max_processes=2
# Loop over the array and run each script in the background
processes=0

for d in "${datasets[@]}";do
    for lr in "${lrs[@]}";do
        for wd in "${wds[@]}";do
            for ep in "${eps[@]}";do
                for m in "${margin[@]}";do
                    for b in "${bath_sizes[@]}";do
                        for dr in "${dropouts[@]}";do
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
                            --epoch_decay $ep \
                            --margin $m \
                            --loss auc \
                            --augmentations basic \
                            --aug_args '' \
                            --dropout $dr > "${processes}_auc.txt" &

                            sleep 1

                            processes=$((processes + 1))

                            if (( processes >= max_processes )); then
                                wait
                                processes=0
                            fi
                        done
                    done
                done
            done
        done
    done
done
