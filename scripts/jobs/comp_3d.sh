#!/bin/bash

name=lr_aug_batch
datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

lrs=(1e-1 1e-2)
bath_sizes=(64)
wds=(1e-4 1e-5)
eps=(3e-2)
b1s=(0.9 0.95)
b2s=(0.95 0.99)
margin=(1.0 0.6)
dropouts=(0)

counter=0
d=$1

# for d in "${datasets[@]}";do
    # touch "../jobs3d${d}.txt"

for lr in "${lrs[@]}";do
    for wd in "${wds[@]}";do
        for ep in "${eps[@]}";do
            for b1 in "${b1s[@]}"; do
                for b2 in "${b2s[@]}"; do
                    for m in "${margin[@]}";do
                        for b in "${bath_sizes[@]}";do
                            for dr in "${dropouts[@]}";do
                                # echo "./jobs/job2d_auc.sh ${name} $(uuidgen) ${d} ${b} ${lr} ${wd} ${ep} ${m} ${dr}" >> "../jobs2d${d}.txt"
                                echo Starting counter $counter
                                ./scripts/jobs/job3d_comp.sh $name $(uuidgen) $d $b $lr $wd $ep $b1 $b2 $m $dr
                                counter=$((counter+1))
                            done
                        done
                    done
                done
            done
        done
    done
done

