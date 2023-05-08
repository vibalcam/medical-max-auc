#!/bin/bash

name=aug_hp
# datasets=("nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

augs=(ra gb et cj rc)

for au in "${augs[@]}";do
    ./$1 $name $(uuidgen) $au
done
