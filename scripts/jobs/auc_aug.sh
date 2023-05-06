#!/bin/bash

name=lr_aug_batch
datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

augs=(ra ra.gb ra.et ra.gb.cj ra.et.cj)

counter=0
d=$1


for au in "${augs[@]}";do
    ./$1 $name $(uuidgen) $d $b $lr $wd $ep $m $dr
done
