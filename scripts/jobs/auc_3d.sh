#!/bin/bash

name=lr_aug_batch
datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")
lrs=(1e-1 1e-2)
bath_sizes=(32)
wds=(1e-4 1e-5)
eps=(3e-2 3e-3)
margin=(1.0 0.6)


# augs=(ra ra.gb ra.et ra.gb.cj ra.et.cj)

counter=0
d=$1

# for d in "${datasets[@]}";do
    # touch "../jobs3d${d}.txt"

for lr in "${lrs[@]}";do
    for wd in "${wds[@]}";do
        for ep in "${eps[@]}";do
            for m in "${margin[@]}";do
                for b in "${bath_sizes[@]}";do

                        echo Starting counter $counter
                        ./scripts/jobs/job3d_auc.sh $name $(uuidgen) $d $b $lr $wd $ep $m $dr

                    done
                done
            done
        done
    done
done
