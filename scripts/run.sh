#!/bin/bash

workers=6
saved_folder='./saved_models'
# seed=1234

## pauc
# lr {1e-3, 1e-4, 1e-5}, 5e-6
# wd 2e-4, 1e-2
# gammas {0.9, 0.5}
# lambda, tau {0.1, 1, 10}
python train_torch.py \
    --arch attention \
    --learning-rate 1e-3 \
    --gammas 0.9 0.9 \
    --lbda 0.5 \
    --tau 1.0 \
    --weight-decay 2e-4 \
    --epochs 20 \
    --batch-size 16 \
    --optimizer adamw \
    --loss_type pauc \
    --warmup-epochs 0.5 \
    --save_dir $saved_folder \
    --workers $workers \
    --evaluate_every 0.25 \
    --save_every_epochs 1 \
    --early_stopping_patience 100 \
    --save 'defender/att.pkl' \
    --hidden_dim 752 \
    --n_layers 12 \
    --num_heads 16 \
    --dropout_rate 0.2 \
    --att_dropout 0.2
    # --pretrained 'saved_models/attention/bce/[0, 1, 2, 3]/version_5/best_auc.ckpt'
    # --resume 'saved_models/attention/bce/[0, 1, 2, 3]/version_4/best_auc.ckpt'
    # --seed 123456
    # --use_features

# python train_torch.py \
#     --arch attention \
#     --learning-rate 5e-6 \
#     --gammas 0.9 0.9 \
#     --lbda 0.5 \
#     --tau 10.0 \
#     --weight-decay 2e-4 \
#     --epochs 10 \
#     --batch-size 32 \
#     --optimizer 'adamw' \
#     --loss_type bce \
#     --warmup-epochs 0.5 \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --evaluate_every 0.25 \
#     --save_every_epochs 1 \
#     --early_stopping_patience 100 \
#     --save 'defender/att.pkl' \
#     --hidden_dim 512 \
#     --n_layers 12 \
#     --num_heads 4 \
#     --dropout_rate 0.5 \
#     --att_dropout 0.5
#     # --seed 123456
#     # --use_features



# python train_torch.py \
#     --arch ff \
#     --lr 1e-5 \
#     --epochs 400 \
#     --batch-size 4000 \
#     --optimizer 'adamw' \
#     --loss_type pauc \
#     --warmup-epochs 40 \
#     --save_dir $saved_folder \
#     --workers $workers \
#     --evaluate_every 1 \
#     --save_every_epochs 20 \
#     --early_stopping_patience 100 \
#     --save 'defender/att_ember.pkl' \
#     --hidden_dim 1024 1024 1024 1024 \
#     --num_heads 12 \
#     --dropout_rate 0.1 \
#     --att_dropout 0.1
#     # --use_features

# python test.py -m 'defender/att.pkl'
