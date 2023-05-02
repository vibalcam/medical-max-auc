import numpy as np
import torch
from torchvision import transforms as T

from models import loader

def basic(ndim, args, mean=0.5, std=0.5):
    if ndim <= 3:
        # output should be: N, C, H, W
        train_transform = T.Compose([
            # T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean,std),
        ])

        eval_transform = T.Compose([
            # T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean,std),
        ])
    elif ndim == 4:
        # output should be: N, C, D, H, W
        train_transform = T.Compose([
            # lambda x: x.astype(np.float32),
            lambda x: torch.from_numpy(np.swapaxes(x, 1, 3)).float(),
            T.Normalize(mean,std),
        ])

        eval_transform = T.Compose([
            # lambda x: x.astype(np.float32),
            lambda x: torch.from_numpy(np.swapaxes(x, 1, 3)).float(),
            T.Normalize(mean,std),
        ])
    else:
        raise NotImplementedError()

    return train_transform, eval_transform


def convirt(ndim, args, mean=0.5, std=0.5):
    if ndim <= 3:
        # output should be: N, C, H, W
        to_tensor = T.ToTensor()
    elif ndim == 4:
        # output should be: N, C, D, H, W
        to_tensor = lambda x: torch.from_numpy(np.swapaxes(x, 1, 3)).float()
    else:
        raise NotImplementedError()
    
    d_transf = {
        'h': T.RandomHorizontalFlip(),
        'ra': T.RandomApply([
                T.RandomAffine(
                    degrees=(-5,5),
                    # translate=(0.01,0.01),
                    # scale=(0.97,1.02),
                    fill=128,
                ),
            ], p=0.5),
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
        'rc': T.RandomResizedCrop(
                (28,28),
                scale=(0.8,1),
            ),
    }
    
    l_aug = []
    for k in args.aug_args.split('.'):
        l_aug.append(d_transf[k])
    
    if ndim == 4:
        l_aug.insert(0, to_tensor)
    else:
        l_aug.append(to_tensor)

    l_aug.append(T.Normalize(mean, std))
    train_transform = T.Compose(l_aug)

    eval_transform = T.Compose([
        # T.Grayscale(3),
        to_tensor,
        T.Normalize(mean, std),
    ])

    return train_transform, eval_transform
