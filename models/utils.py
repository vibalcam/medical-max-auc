import argparse
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import pathlib
from torch.utils.data import random_split
import pandas as pd
import glob
from scipy.sparse import csr_matrix
import torch.nn as nn
import numpy as np
from sklearn import metrics
import argparse


def get_logger_name(args: argparse.Namespace) -> str:
    """
    Generates a logger name based on the first two letters of each parameter name in the argparse object.
    
    Args:
        args: An argparse.Namespace object containing the parsed command-line arguments.
        
    Returns:
        A string representing the logger name.
    """
    # Construct the logger name from the parsed arguments
    logger_name = ""
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, list):
            value = "_".join([str(k) for k in value])
        logger_name += f"{arg[:2]}={value}_"
    logger_name = logger_name[:-1]  # Remove the trailing underscore
    
    return logger_name


class ClassificationMetrics:
    def __init__(self, prefix='') -> None:
        self.target = []
        self.preds = []
        self.prefix = prefix

    def update(self, target: torch.Tensor, preds: torch.Tensor) -> Dict[str, float]:
        target = target.detach().clone().cpu()
        preds = preds.detach().clone().cpu()
        self.target.append(target)
        self.preds.append(preds)

        return self._get_results(target, preds)

    def compute(self) -> Dict[str, float]:
        target = torch.cat(self.target, dim=0)
        preds = torch.cat(self.preds, dim=0)
        
        return self._get_results(target, preds)

    def __len__(self) -> int:
        return len(self.target)
    
    def _get_results(self, target, preds):
        y_pred = (preds >= 0.5).int()

        acc = metrics.accuracy_score(target, y_pred)
        auc = []
        for k in range(target.shape[1]):
            if len(np.unique(target[:,k])) == 2:
                auc.append(metrics.roc_auc_score(target[:,k], preds[:,k]))
            elif len(np.unique(target[:,k])) > 2:
                raise NotImplementedError()
        
        if len(auc) > 0:
            auc = np.mean(auc)
        else:
            auc = 0.5

        try:
            auc_test = metrics.roc_auc_score(target, preds)
            assert np.isclose(auc_test, auc)
        except:
            pass

        return {
            self.prefix + 'acc': acc,
            self.prefix + 'auc': auc,
        }
    
    def reset(self) -> None:
        self.target = []
        self.preds = []


def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model


def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children(): 
        if hasattr(nn, child.__class__.__name__) and \
            'batchnorm' in child.__class__.__name__.lower():
            TargetClass = globals()['Synchronized'+child.__class__.__name__]
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)


def plot_roc(tpr,fpr):
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    return fig


def split_arrays(
    arrays:List[Union[np.array, csr_matrix]],
    lengths,
    return_idx : bool = False,
    seed : int = None,
):
    # get lengths for datasets
    n_samples = arrays[0].shape[0]
    split_length = np.floor(np.cumsum(lengths) * n_samples).astype(int)
    split_length[-1] = n_samples

    # shuffle indices
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(n_samples)

    # split dataset
    splits = []
    splits_idx = []
    start = 0
    for l in split_length:
        arr_idx = idx[start:l]
        arrs = [k[arr_idx] for k in arrays]
        start = l
        splits.append(arrs)
        splits_idx.append(arr_idx)

    return (splits, splits_idx) if return_idx else splits


def split_generator_dataset(
    dataset,
    lengths,
    seed: int,
):
    # get lengths for datasets
    lengths = np.floor(np.asarray(lengths) * len(dataset))
    lengths[-1] = len(dataset) - np.sum(lengths[:-1])
    # random split
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    subsets = random_split(
        dataset, 
        lengths.astype(int).tolist(),
        generator,
    )
    return subsets


def create_batches(arr, batch_size=5000):
    start = 0
    for k in range(int(np.ceil(arr.shape[0] / batch_size))):
        if start+batch_size >= arr.shape[0]:
            batch_size = arr.shape[0] - start
        yield arr[start:start+batch_size,...], start, start+batch_size
        start += batch_size


def batched_fit_transform(x, transform, batch_size=5000, combine_last=False, ignore_last=False):
    '''
    Fits the data to the transformation and then transforms it in batches
    '''
    if combine_last and ignore_last:
        raise Exception("Cannot combine_last and ignore_last at the same time: choose one")

    # calculate index of last batch
    if x.shape[0] % batch_size != 0 and (combine_last or ignore_last):
        idx_last = (x.shape[0] // batch_size - 1) * batch_size
    else:
        idx_last = x.shape[0]

    # do partial fits in batches
    for arr,start,end in create_batches(x[:idx_last, ...], batch_size=batch_size):
        transform.partial_fit(arr)
    # do partial fit on extra data (size lower than batch_size)
    if combine_last:
        transform.partial_fit(x[idx_last+1:, ...])
    elif ignore_last:
        transform.partial_fit(x[idx_last+1:idx_last+1+batch_size, ...])

    # transform data
    for arr,start,end in create_batches(x, batch_size=batch_size):
        out = transform.transform(arr)
        x[start:end,:out.shape[1]] = out

    return x[:,:out.shape[1]]


# ----------------------------------------------------------------------------------


MODEL_CLASS_KEY = 'model_class'
FOLDER_PATH_KEY = 'path_name'

# to dynamically import
# import importlib
# module = importlib.import_module(module_name)
# class_ = getattr(module, class_name)
# instance = class_()


def save_model(
        model: torch.nn.Module,
        folder: Union[pathlib.Path, str],
        model_name: str,
        models_dict:Dict,
        param_dicts: Dict = None,
        save_model: bool = True,
) -> None:
    """
    Saves the model so it can be loaded after

    :param model_name: name of the model to be saved (non including extension)
    :param folder: path of the folder where to save the model
    :param param_dicts: dictionary of the model parameters that can later be used to load it
    :param model: model to be saved
    :param save_model: If true the model and dictionary will be saved, otherwise only the dictionary will be saved
    """
    # create folder if it does not exist
    folder_path = f"{folder}/{model_name}"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    # save model
    if save_model:
        torch.save(model.state_dict(), f"{folder_path}/{model_name}.th")

    # save dict
    if param_dicts is None:
        param_dicts = {}

    # get class of the model
    model_class = None
    for k, v in models_dict.items():
        if isinstance(model, v):
            model_class = k
            break
    if model_class is None:
        raise Exception("Model class unknown")
    param_dicts[MODEL_CLASS_KEY] = model_class

    # save the dictionary as plain text and pickle
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict", as_str=True)
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict.pickle", as_str=False)


def load_model(
    folder_path: str, 
    models_dict:Dict,
    model_class: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)

    :param folder_path: folder path of the model to be loaded
    :param model_class: one of the model classes in `models_dict` dict. If none, it is obtained from the dictionary
    :return: the loaded model and the dictionary of parameters
    """
    folder_path = pathlib.Path(folder_path)
    path = f"{folder_path.absolute()}/{folder_path.name}"
    # use pickle dictionary
    dict_model = load_dict(f"{path}.dict.pickle")

    # get model class
    if model_class is None:
        model_class = dict_model.get(MODEL_CLASS_KEY)

    # set folder path
    dict_model[FOLDER_PATH_KEY] = str(folder_path)

    return load_model_data(models_dict[model_class](**dict_model), f"{path}.th"), dict_model


def load_model_data(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Loads a model than has been previously saved

    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


# ----------------------------------------------------------------------------------


def save_pickle(obj, path: Union[str, Path]):
    """
    Saves an object with pickle

    :param obj: object to be saved
    :param save_path: path to the file where it will be saved
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]):
    """
    Loads an object with pickle from a file

    :param path: path to the file where the object is stored
    :param cls: The class to use for unpickling
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def dict_to_str(d: Dict):
    str = '{\n'
    for k,v in d.items():
        if isinstance(v, dict):
            v = dict_to_str(v)
        str += f"'{k}':{v},\n"
    return str + '}'


def save_dict(d: Dict, path: str, as_str: bool = False) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    :param as_str: If true, it will save as a string. If false, it will use pickle
    """
    if as_str:
        with open(path, 'w', encoding="utf-8") as file:
            # file.write(str(d))
            file.write(dict_to_str(d))
    else:
        save_pickle(d, path)


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file (plain text or pickle)

    :param path: path where the dictionary was saved
    :return: the loaded dictionary
    """
    try:
        return load_pickle(path)
    except pickle.UnpicklingError as e:
        # print(e)
        pass

    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        s = file.read()
        return dict(literal_eval(s))


def log(s):
    """
    Helper method to log

    Parameters
    ----------
    s : str
        string to log
    """
    print(s)


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    :param seed: seed for the random generators
    """
    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # make deterministic
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        # for deterministic behavior on cuda >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self): 
        return self.__dict__
    def __setstate__(self, d): 
        self.__dict__.update(d)

    def copy(self):
        return dotdict(super().copy())


def datetime_to_timestamp(dt):
    return np.array(dt, dtype='datetime64').view('int64') * 1e-9
