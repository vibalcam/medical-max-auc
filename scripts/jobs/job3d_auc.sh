#!/bin/bash

##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            	#Do not propagate environment
#SBATCH --get-user-env=L         	#Replicate login environment
    
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=med
#SBATCH --time=2:00:00 
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --output=logs/%j.txt  	#Send stdout/err
#SBATCH --gres=gpu:1             	#Request 1 GPU per node can be 1 or 2 (gpu:a100:1)
#SBATCH --partition=gpu          	#Request the GPU partition/queue

# squeue -u vibalcam
# sbatch jobPytorch.sh
# scancel 6565
# seff 6591423

source prepare.sh || exit 1

workers=$SLURM_CPUS_PER_TASK
id=$SLURM_JOBID
saved_folder='./saved_models'

name=$1
d=$2
b=$3
lr=$4
wd=$5
ep=$6
m=$7
dr=$8

echo "Starting $name $id with $workers workers"

python train.py \
    --name $name \
    --dataset $d \
    --save_dir $saved_folder \
    --workers $workers \
    --seed 123456 \
    --epochs 200 \
    --lr_steps 100 150 \
    --batch_size $b \
    --lr $lr \
    --weight_decay $wd \
    --epoch_decay $ep \
    --margin $m \
    --loss auc \
    --augmentations basic \
    --aug_args '' \
    --dropout $dr \
    --type_3d '3d'
