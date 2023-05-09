import os
from models.utils import dotdict

import copy

import os
import glob
from train import Module, main
from arguments import args
from arguments import args
from pandas import json_normalize

folder = 'other'
# paths = glob.glob(os.path.join(folder, "**/*.ckpt"), recursive=True)
paths = glob.glob(os.path.join(folder, "**/best.ckpt"), recursive=True)

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

df = json_normalize(results)

df = df[['dataset', 'test.test_auc']]
df['test.test_auc'] = df['test.test_auc'].round(3)

print('*' * 50)
print('*' * 50)
print(df)
print('*' * 50)
print('*' * 50)
