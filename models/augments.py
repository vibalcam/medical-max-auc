import numpy as np
import torch
from torchvision import transforms as T

from models import loader

def basic(ndim, args):
    if ndim <= 3:
        # output should be: N, C, H, W
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])

        eval_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])
    elif ndim == 4:
        # output should be: N, C, D, H, W
        train_transform = T.Compose([
            lambda x: torch.from_numpy(np.swapaxes(x, 1, 3)).float(),
            T.Normalize(0.5, 0.5),
        ])

        eval_transform = T.Compose([
            lambda x: torch.from_numpy(np.swapaxes(x, 1, 3)).float(),
            T.Normalize(0.5, 0.5),
        ])
    else:
        raise NotImplementedError()

    return train_transform, eval_transform


def convirt(ndim, args):
    if ndim <= 3:
        # output should be: N, C, H, W
        to_tensor = T.ToTensor()
    elif ndim == 4:
        # output should be: N, C, D, H, W
        to_tensor = lambda x: torch.from_numpy(np.swapaxes(x, 1, 3))
    else:
        raise NotImplementedError()
    
    # output should be: N, C, H, W
    train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomApply([
            T.RandomAffine(
                degrees=(-15,15),
                translate=(0.1,0.1),
                scale=(0.95,1.05),
                fill=128,
            ),
        ], p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
        ], p=0.8),
        T.RandomApply([
            loader.GaussianBlur((0.1,3.0)),
        ], p=0.5),
        T.RandomApply([
            T.ElasticTransform(alpha=40, sigma=10)
        ], p=0.5),
        to_tensor,
        T.Normalize(mean=0.5, std=0.5),
    ])

    eval_transform = T.Compose([
        to_tensor,
        T.Normalize(0.5, 0.5),
    ])

    return train_transform, eval_transform
