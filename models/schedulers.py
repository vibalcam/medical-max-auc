
import math
import numpy as np


# def adjust_learning_rate(optimizer, init_lr, epoch, args, min_lr=0):
#     """Decays the learning rate with half-cycle cosine after warmup"""
#     warmup_epochs = args.warmup_epochs

#     if epoch < warmup_epochs:
#         lr = (init_lr-min_lr) * epoch / warmup_epochs  + min_lr
#     else:
#         lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
#     for param_group in optimizer.param_groups:
#         param_group[self.key] = lr
#     return lr


class LrScheduler:
    def __init__(self, optimizer, init_lr, key='lr', **kwargs):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.key = key

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group[self.key] = lr
        return lr
    
    def factor_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            lr = param_group[self.key] * factor
            param_group[self.key] = lr
        return lr
    
    def step(self, epoch, **kwargs):
        pass

    def update(self, metric, **kwargs):
        pass


class SchedulerCollection:
    def __init__(self, schedulers=None):
        if schedulers is None:
            schedulers = []
        self.schedulers = schedulers

    def add(self, scheduler):
        self.schedulers.append(scheduler)

    def step(self, epoch):
        for scheduler in self.schedulers:
            scheduler.step(epoch)
    
    def update(self, metric):
        for scheduler in self.schedulers:
            scheduler.update(metric)


class CosineSchedulerWithWarmup(LrScheduler):
    def __init__(self, optimizer, init_lr, warmup_epochs, n_epochs, min_lr=0, **kwargs):
        super().__init__(optimizer, init_lr, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.n_epochs = n_epochs

        # initialize lr
        self.step(0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = (self.init_lr-self.min_lr) * (epoch+1) / self.warmup_epochs + self.min_lr
        else:
            lr = (self.init_lr-self.min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.n_epochs - self.warmup_epochs))) + self.min_lr

        self.update_lr(lr)


class SchedulerMetricWithWarmup(LrScheduler):
    def __init__(self, optimizer, init_lr, warmup_epochs, factor=0.1, min_lr=0, patience=10, maximize=True, **kwargs):
        super().__init__(optimizer, init_lr, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.factor = factor
        self.ind = 1 if maximize else -1
        self.curr = np.inf
        self.patience = patience
        self.curr_patience = patience
        self.epoch = 0

        # initialize lr
        self.step(0)

    def step(self, epoch):
        self.epoch = epoch
        if epoch < self.warmup_epochs:
            lr = (self.init_lr-self.min_lr) * (epoch+1) / self.warmup_epochs  + self.min_lr
            self.update_lr(lr)

    def update(self, metric):
        if self.epoch < self.warmup_epochs:
            return

        if self.ind * metric >= self.curr * self.ind:
            # better metric
            self.curr_patience = self.patience
            self.curr = metric
        else:
            # worse metric
            self.curr_patience -= 1
            print(f"Metric did not improve: {self.curr_patience} steps left to reduce lr")

        if self.curr_patience == 0:
            self.curr_patience = self.patience
            for param_group in self.optimizer.param_groups:
                param_group[self.key] *= self.factor
                if param_group[self.key] < self.min_lr:
                    param_group[self.key] = self.min_lr
                    print(f"Minimum lr {self.min_lr} reached")
                else:
                    print(f"New lr: {param_group[self.key]}")


class StepSchedulerWithWarmup(LrScheduler):
    def __init__(self, optimizer, init_lr, steps, warmup_epochs, factor=0.1, min_lr=0, **kwargs):
        super().__init__(optimizer, init_lr, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.factor = factor
        self.steps = steps

        # initialize lr
        self.step(0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = (self.init_lr-self.min_lr) * (epoch+1) / self.warmup_epochs  + self.min_lr
            self.update_lr(lr)
        elif epoch in self.steps:
            for param_group in self.optimizer.param_groups:
                lr = max(param_group[self.key] * self.factor, self.min_lr)
                param_group[self.key] = lr
