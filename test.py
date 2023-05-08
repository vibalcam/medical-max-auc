import argparse
from functools import partial
import gc
import math
import os
from typing import Any, Optional
import warnings
from matplotlib import pyplot as plt
from models.pretraining import similarity_loss
from models.schedulers import CosineSchedulerWithWarmup, SchedulerCollection, StepSchedulerWithWarmup
from models.utils import ClassificationMetrics, dotdict, get_logger_name, model_to_syncbn, save_dict, save_pickle, set_seed

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import copy

from libauc.losses import AUCMLoss, CompositionalAUCLoss, AUCM_MultiLabel, AUCM_MultiLabel_V2
from libauc import optimizers
import medmnist 
from models import augments
from medmnist import INFO
from models.sampler import DualSampler
import os
import glob
from train import Module, main
from arguments import args
import pandas as pd
from arguments import args

folder = 'best_models'
paths = glob.glob(os.path.join(folder, "**/*.ckpt"), recursive=True)
# paths = glob.glob(os.path.join(folder, "**/last.ckpt"), recursive=True)

results = []
args = vars(args)
for p in paths:
    m = Module.load_from_checkpoint(p)
    a = copy.deepcopy(args)
    a.update(vars(m.hparams.args))
    a = dotdict(a)
    a.test = p
    a.workers = 8

    r = main(a)
    r['dataset'] = a.dataset
    results.append(r)

df = pd.DataFrame(results)

print(df)




