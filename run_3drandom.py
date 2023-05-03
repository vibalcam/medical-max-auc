import random
import subprocess
from train import main
from arguments import args


num_trials = 25
datasets = ["nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d"]


lrs = [1e-1, 1e-2, 1e-3]
batch_sizes = [32, 64]
wds = [1e-3, 1e-4, 1e-5]
eps = [3e-2, 3e-3, 3e-4]
margin = [1.0, 0.8, 0.6]
dropouts = [0, 0.1]
augmentations = []


from itertools import chain, combinations
# Get all possible combinations of strings (1 to len(strings) strings in each combination)
combos = chain.from_iterable(combinations(augmentations, r) for r in range(1, len(augmentations) + 1))
# Join the strings with dots to create a list of all possible concatenated combinations
dot_combos = ['.'.join(c) for c in combos]
# Add the individual strings to the list
result = augmentations + dot_combos + [None]


for d in datasets:
    args.dataset = d

    for i in range(num_trials):
        print(f"Trial {i} of {num_trials} for {d}")
        
        random.seed(123456)
        for k in range(i):
            random.randint(1,num_trials)

        lr = random.choice(lrs)
        b = random.choice(batch_sizes)
        wd = random.choice(wds)
        drop = random.choice(dropouts)
        ep = random.choice(eps)
        m = random.choice(margin)
        aug = random.choice(result)

        print(lr)

        print(args.name)
        args.lr = lr
        args.batch_size = b
        args.dropout = drop
        args.weight_decay = wd
        args.epoch_decay = ep
        args.margin = m
        if aug is None:
            args.augmentations = 'basic'
        else:
            args.augmentations = 'convirt'
            args.aug_args = aug

        main(args)
