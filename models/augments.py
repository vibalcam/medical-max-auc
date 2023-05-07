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
        'h': T.RandomHorizontalFlip(),
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
        'rc': T.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.8,1),
            ),
    }
    
    train_transform = []
    for k in args.aug_args.split('.'):
        if ndim == 4 and k == 'gb':
            continue

        train_transform.append(d_transf[k])
    
    train_transform.insert(0, to_tensor)
    # if ndim == 4:
    #     train_transform.insert(0, to_tensor)
    # else:
    #     train_transform.append(to_tensor)

    train_transform.append(T.Normalize(mean, std))

    eval_transform = [
        # T.Grayscale(3),
        to_tensor,
        T.Normalize(mean, std),
    ]

    if args.resize is not None:
        train_transform.insert(
            1, T.Resize(args.resize, antialias=True)
        )
        eval_transform.insert(
            1, T.Resize(args.resize, antialias=True)
        )

    return T.Compose(train_transform), T.Compose(eval_transform)
