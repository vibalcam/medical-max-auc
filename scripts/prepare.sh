#!/bin/bash

# modules needed for running DL jobs. Module restore will also work
# ml GCCcore/11.2.0 CUDAcore/11.0.2
# ml Python/3.9.6
# ml GCC/10.3.0  OpenMPI/4.1.1
# ml PyTorch/1.12.1
# torch.cuda.is_available()

ml GCCcore/11.2.0 CUDAcore/11.0.2
ml Anaconda3/2022.10
cd $SCRATCH
source venv/bin/activate

# scripts or executables
cd /scratch/user/vibalcam/medical-auc
