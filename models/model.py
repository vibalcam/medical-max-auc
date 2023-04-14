from typing import Any, List
import warnings
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, roc_auc_score
import numpy as np
import torch
from models.utils import save_pickle, load_pickle
import torchmetrics as metrics

from torchmetrics.functional.classification import binary_roc
from torchmetrics import Metric
import torch.nn as nn


# todo! finish
class Model(nn.Module):
    pass
