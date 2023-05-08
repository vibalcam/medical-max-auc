!/bin/bash

workers=8

saved_folder='./tmp'
log_folder='./logs_txt'

datasets=("breastmnist" "pneumoniamnist")
# datasets=("breastmnist")
#  "nodulemnist3d" "adrenalmnist3d" "vesselmnist3d" "synapsemnist3d")

# loop through indices of datasets
for (( i=0; i<${#datasets[@]}; i++ ))
do

    d="${datasets[$i]}"

    # basic training
    echo "Starting basic training for dataset $d"
    python3 train_ensemble.py \
        --name default \
        --dataset $d \
        --save_dir $saved_folder \
        --workers $workers \
        --seed 123456 \
        --epochs 200 \
        --lr_steps 50 75 \
        --batch_size 128 \
        --lr 1e-1 \
        --weight_decay 1e-4 \
        --epoch_decay 3e-2 \
        --margin 0.6 \
        --loss auc \
        --train_on_val "true" \
        --early_stopping_patience 10
done
        # --augmentations "convirt"\
        # --aug_args "h.ra.et"\
        # --oversample "true" \

# workers=8
# saved_folder='./ensemble_new'

# d=breastmnist
# # ["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",]

# name=$d

# ## best val auc
# python train.py \
#     --name auc_val \
#     --dataset $d \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --seed 123456 \
#     --epochs 200 \
#     --lr_steps 100 150 \
#     --batch_size 128 \
#     --lr 0.1 \
#     --weight_decay 1e-5 \
#     --epoch_decay 0.03 \
#     --margin 1.0 \
#     --loss auc \
#     --augmentations basic \
#     --aug_args '' \
#     --dropout 0