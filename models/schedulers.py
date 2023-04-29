
import numpy as np


def adjust_learning_rate(optimizer, init_lr, epoch, args, min_lr=0):
    """Decays the learning rate with half-cycle cosine after warmup"""
    warmup_epochs = args.warmup_epochs

    if epoch < warmup_epochs:
        lr = (init_lr-min_lr) * epoch / warmup_epochs  + min_lr
    else:
        lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class LrScheduler:
    def __init__(self, optimizer, init_lr, **kwargs):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def factor_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] * factor
            param_group['lr'] = lr
        return lr
    
    def step(self, epoch, **kwargs):
        pass

    def update(self, metric, **kwargs):
        pass


class SchedulerMetricWithWarmup(LrScheduler):
    def __init__(self, optimizer, init_lr, warmup_epochs, factor=0.1, min_lr=0, patience=10, maximize=True):
        super().__init__(optimizer, init_lr)
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
            lr = (self.init_lr-self.min_lr) * epoch / self.warmup_epochs  + self.min_lr
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
                param_group['lr'] *= self.factor
                if param_group['lr'] < self.min_lr:
                    param_group['lr'] = self.min_lr
                    print(f"Minimum lr {self.min_lr} reached")
                else:
                    print(f"New lr: {param_group['lr']}")


class StepSchedulerWithWarmup(LrScheduler):
    def __init__(self, optimizer, init_lr, steps, warmup_epochs, factor=0.1, min_lr=0):
        super().__init__(optimizer, init_lr)
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.factor = factor
        self.steps = steps

        # initialize lr
        self.step(0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = (self.init_lr-self.min_lr) * epoch / self.warmup_epochs  + self.min_lr
            self.update_lr(lr)
        elif epoch in self.steps:
            for param_group in self.optimizer.param_groups:
                lr = max(param_group['lr'] * self.factor, self.min_lr)
                param_group['lr'] = lr
