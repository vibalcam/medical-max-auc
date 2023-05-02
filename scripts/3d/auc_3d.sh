#!/bin/bash

name=lr_aug_batch
datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")
lrs=(1e-1 1e-2 1e-3)
bath_sizes=(32 64)
wds=(1e-3 1e-4 1e-5)
eps=(3e-2 3e-3 3e-4)
margin=(1.0 0.8 0.6)
dropouts=(0 0.1)

# augs=(ra ra.gb ra.et ra.gb.cj ra.et.cj)

for d in "${datasets[@]}";do
    for lr in "${lrs[@]}";do
        for wd in "${wds[@]}";do
            for ep in "${eps[@]}";do
                for m in "${margin[@]}";do
                    for b in "${bath_sizes[@]}";do
                        for dr in "${dropouts[@]}";do
                            echo $d $lr $wd $ep $m $b $dr
                            sbatch ../jobs/job3d_auc.sh $name $d $b $lr $wd $ep $m $dr
                        done
                    done
                done
            done
        done
    done
done
