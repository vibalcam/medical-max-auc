#!/bin/bash

# rsync -avzP --ignore-existing --exclude='.git' --exclude='logs_txt' --exclude='tmp' --exclude='tmp_calc' --exclude='*.ckpt' --exclude='data_test' --exclude='data_features_combined' -e ssh terra:/scratch/user/vibalcam/medical-auc/. ../.

# rsync -avzP --delete --update --exclude='.git' --exclude='logs_bce' --exclude='logs_txt' --exclude='tmp' --exclude='tmp_saved_models' --exclude='saved_models' -e ssh ../. terra:/scratch/user/vibalcam/medical-auc/.
