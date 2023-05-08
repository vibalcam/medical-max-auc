#!/bin/bash

name=aug_hp
# datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

## 3d
augs=(ra et rc m)

for au in "${augs[@]}";do
    ./$1 $au
done
