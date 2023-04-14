import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import os
import warnings
import glob
import pandas as pd

from torch.utils.data import Dataset


class PEDataset(Dataset):
    def __init__(self, data_path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
