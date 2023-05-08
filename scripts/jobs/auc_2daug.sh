#!/bin/bash

name=aug_hp
# datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

## 2d
augs=(ra cj gb et rc)

for au in "${augs[@]}";do
    ./$1 $au
done
