#!/bin/bash

##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            	#Do not propagate environment
#SBATCH --get-user-env=L         	#Replicate login environment
    
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=med
#SBATCH --time=3:00:00 
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=36
#SBATCH --mem=8GB
#SBATCH --output=logs/%j.txt  	#Send stdout/err
#SBATCH --gres=gpu:1             	#Request 1 GPU per node can be 1 or 2 (gpu:a100:1)
#SBATCH --partition=gpu          	#Request the GPU partition/queue

# squeue -u vibalcam
# sbatch jobPytorch.sh
# scancel 6565
# seff 6591423

tamulauncher --commands-pernode 4 $1
