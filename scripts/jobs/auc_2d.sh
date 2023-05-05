#!/bin/bash

name=lr_aug_batch
datasets=("breastmnist" "pneumoniamnist" "chestmnist")

# lrs=(1e-1)
# bath_sizes=(128)
# wds=(1e-3 1e-4 1e-5)
# eps=(3e-2)
# margin=(1.0 0.8 0.6)

lrs=(1e-2)
bath_sizes=(128)
wds=(1e-4 1e-5)
eps=(3e-2)
margin=(1.0 0.6)
dropouts=(0 0.1)

# augs=(ra ra.gb ra.et ra.gb.cj ra.et.cj)

counter=0
d=$1

# for d in "${datasets[@]}";do
#     touch "../jobs2d${d}.txt"
    
for lr in "${lrs[@]}";do
    for wd in "${wds[@]}";do
        for ep in "${eps[@]}";do
            for m in "${margin[@]}";do
                for b in "${bath_sizes[@]}";do
                    for dr in "${dropouts[@]}";do
                        # echo "./jobs/job2d_auc.sh ${name} $(uuidgen) ${d} ${b} ${lr} ${wd} ${ep} ${m} ${dr}" >> "../jobs2d${d}.txt"

                        echo Starting counter $counter
                        ./scripts/jobs/job2d_auc.sh $name $(uuidgen) $d $b $lr $wd $ep $m $dr

                        counter=$((counter+1))
                    done
                done
            done
        done
    done
done
# done

# echo "Total jobs: ${counter}"
