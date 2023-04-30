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
    
    train_transform = T.Compose([
        # T.Grayscale(3),
        # T.RandomHorizontalFlip(0.5),
        T.RandomApply([
            T.RandomAffine(
                degrees=(-5,5),
                # translate=(0.1,0.1),
                # scale=(0.97,1.02),
                fill=128,
            ),
        ], p=0.5),
        # T.RandomApply([
        #     T.ColorJitter(brightness=(0.01, 0.05), contrast=(0.01, 0.05)),
        # ], p=0.5),
        T.RandomApply([
            loader.GaussianBlur((0.1,0.3)),
            # T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))
        ], p=0.2),
        # T.RandomApply([
        #     T.ElasticTransform(
        #         alpha=0.5, 
        #         sigma=0.5,
        #         fill=128,
        #     )
        # ], p=0.2),
        to_tensor,
        T.Normalize(mean=0.5, std=0.5),
    ])

    eval_transform = T.Compose([
        # T.Grayscale(3),
        to_tensor,
        T.Normalize(0.5, 0.5),
    ])

    return train_transform, eval_transform
