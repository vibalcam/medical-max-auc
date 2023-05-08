import warnings
import numpy as np
import torch
from torchvision import transforms as T

from models import loader

def basic(ndim, args, mean=0.5, std=0.5):
    if ndim <= 3:
        # output should be: N, C, H, W
        to_tensor = T.ToTensor()
    elif ndim == 4:
        # output should be: N, C, D, H, W
        to_tensor = lambda x: torch.from_numpy(np.swapaxes(x, 1, 3)).float()
    else:
        raise NotImplementedError()
    
    train_transform = [
        to_tensor,
        T.Normalize(mean,std),
    ]

    eval_transform = [
        to_tensor,
        T.Normalize(mean,std),
    ]

    if args.resize is not None:
        train_transform.insert(
            1, T.Resize(args.resize, antialias=True)
        )
        eval_transform.insert(
            1, T.Resize(args.resize, antialias=True)
        )

    return T.Compose(train_transform), T.Compose(eval_transform)


def convirt(ndim, args, mean=0.5, std=0.5):
    image_size = 28 if args.resize is None else args.resize

    if ndim <= 3:
        # output should be: N, C, H, W
        to_tensor = T.ToTensor()
    elif ndim == 4:
        # output should be: N, C, D, H, W
        to_tensor = lambda x: torch.from_numpy(np.swapaxes(x, 1, 3)).float()
    else:
        raise NotImplementedError()
    
    d_transf = {
        'ra': T.RandomApply([
                T.RandomAffine(
                    degrees=(-5,5),
                    # translate=(0.01,0.01),
                    # scale=(0.97,1.02),
                    fill=128,
                ),
            ], p=0.2),
        'cj': T.RandomApply([
                # T.ColorJitter(brightness=(0.01, 0.2), contrast=(0.01, 0.2)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ], p=0.2),
        'gb': T.RandomApply([
                loader.GaussianBlur((0.1,0.3)),
            ], p=0.2),
        'et': T.RandomApply([
                T.ElasticTransform(
                    alpha=0.1,
                    sigma=0.1,
                    fill=128,
                )
            ], p=0.2),
        'rc': T.RandomApply([
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.7, 1),
                    antialias=True,
                )
            ], p=0.2),
        'm': [
            lambda x: x * np.random.uniform(),
            lambda x: x * 0.5,
        ]
    }
    
    train_transform = []
    eval_transform = []
    for k in args.aug_args.split('.'):
        if ndim == 4 and k == 'cj':
            warnings.warn('ColorJitter is not supported for 3D images.')
            continue

        t = d_transf[k]
        if isinstance(t, list) or isinstance(t, tuple):
            train_transform.append(t[0])
            eval_transform.append(t[1])
        else:
            train_transform.append(t)
    
    print('==> train_transform', train_transform)

    train_transform.insert(0, to_tensor)
    train_transform.append(T.Normalize(mean, std))

    eval_transform.insert(0, to_tensor)
    eval_transform.append(T.Normalize(mean, std))

    # eval_transform = [
    #     # T.Grayscale(3),
    #     to_tensor,
    #     T.Normalize(mean, std),
    # ]

    if args.resize is not None:
        train_transform.insert(
            1, T.Resize(args.resize, antialias=True)
        )
        eval_transform.insert(
            1, T.Resize(args.resize, antialias=True)
        )

    return T.Compose(train_transform), T.Compose(eval_transform)
