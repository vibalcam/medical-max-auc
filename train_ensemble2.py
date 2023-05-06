import random
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision.models import resnet18
import pytorch_lightning as pl
from train import Module as GeneralModel

import argparse
from functools import partial
import gc
import math
import os
from typing import Any, Optional
import warnings
from matplotlib import pyplot as plt
from models.pretraining import similarity_loss
from models.schedulers import CosineSchedulerWithWarmup, SchedulerCollection, StepSchedulerWithWarmup
from models.utils import ClassificationMetrics, get_logger_name, model_to_syncbn, save_dict, save_pickle, set_seed

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from libauc.losses import AUCMLoss, CompositionalAUCLoss
from libauc import optimizers
import medmnist 
from models import augments
from medmnist import INFO
from models.sampler import DualSampler
from train import Module as GeneralModel


# Define the ResNet18 model
# class ResNet18(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet18, self).__init__()
#         self.resnet18 = resnet18(pretrained=False)
#         num_ftrs = self.resnet18.fc.in_features
#         self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):
#         x = self.resnet18(x)
#         return x

# Define the ensemble model which consists of multiple ResNet18 models trained on random subsets of the train dataset
# class BaggedResNet18Ensemble(pl.LightningModule):
#     def __init__(self, num_classes=10, num_models=3, train_dataset=None):
#         super(BaggedResNet18Ensemble, self).__init__()
#         self.num_models = num_models
#         self.train_dataset = train_dataset
#         self.models = nn.ModuleList([ResNet18(num_classes) for _ in range(num_models)])
#         self.criterion = nn.CrossEntropyLoss()
#         self.accuracy = pl.metrics.Accuracy()

class EnsembleModel(pl.LightningModule):
    # def __init__(self, num_classes=10, num_models=3):
    def __init__(
        self,
        args,
        img_shape,
        num_outputs,
        num_models,
        train_dataset,
        val_dataset,
        **kwargs,
    ):
        super(EnsembleModel, self).__init__()
        self.args = args
        self.num_models = num_models
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.models = nn.ModuleList([GeneralModel(args=args, img_shape=img_shape, num_outputs=num_outputs) for _ in range(num_models)])
        # self.criterion = nn.CrossEntropyLoss()
        # self.accuracy = pl.metrics.Accuracy()
        self.define_metrics()
        self.lr = self.args.lr

        # define loss function
        if self.args.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss_type == 'auc':
            self.criterion = AUCMLoss()
        elif args.loss_type == 'comp':
            self.criterion = CompositionalAUCLoss(last_activation=None)
        elif args.loss_type == 'pre':
            self.criterion = similarity_loss
        else:
            raise NotImplementedError()

    def define_metrics(self):
        self.train_metrics = ClassificationMetrics('train_')
        self.val_metrics = ClassificationMetrics('val_')
        self.test_metrics = ClassificationMetrics('test_')


    def forward(self, x):
        output = 0
        for model in self.models:
            output += model(x)
        return output / self.num_models
        # print(output.shape[0])
        # return 0
        # return torch.zeros(output.shape[0])

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self(x)
        # print("HERE")
        # print("y_hat", y_hat)
        # print("y", y)
        # print(y.shape)
        # loss = self.criterion(y_hat, torch.sigmoid(y))
        # self.log('train_loss', loss)
        # return loss
        x, target = batch
        output = self(x)
        if self.args.loss_type=='auc':
            loss = self.criterion(torch.sigmoid(output), target.float())
        elif self.args.loss_type=='pre':
            loss = self.criterion(
                output, 
                target,
                T=0.1, 
                device=self.device,
            )
        else:
            loss = self.criterion(output, target.float())

        if output.isnan().any():
            warnings.warn("Nan values being generated")
        if loss.isnan():
            warnings.warn("Getting nan loss")

        # calculate metrics
        if self.args.loss_type != 'pre':
            y_pred = torch.sigmoid(output)
            r = self.train_metrics.update(target.int(), y_pred)
            self.log_dict(r)

        # log loss for training step and average loss for epoch
        self.log_dict({
            "train_loss": loss,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)
    #     # acc = self.accuracy(y_hat.argmax(dim=1), y)
    #     acc = self.val_metrics.update(y.int(), torch.sigmoid(y_hat))
    #     self.log('val_loss', loss)
    #     self.log('val_acc', acc)
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):   
        if self.args.loss_type == 'pre':
            return
        # run through model
        x, target = batch[:2]
        output = self(x)

        # calculate metrics
        self.val_metrics.update(target.int(), torch.sigmoid(output))

        if self.lr_scheduler is not None:
            self.lr_scheduler.update(None)

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # return optimizer
        self.lr_scheduler = SchedulerCollection()
        ## optimizer
        if self.args.loss_type == 'auc':
            optimizer = optimizers.PESG(
                self.models, 
                loss_fn=self.criterion, 
                lr=self.lr, 
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                epoch_decay=self.args.epoch_decay,
                margin=self.args.margin,
            )
        elif self.args.loss_type == 'comp':
            optimizer = optimizers.PDSCA(
                self.models, 
                loss_fn=self.criterion, 
                lr=self.args.lr, 
                lr0=self.args.lr0,
                beta1=self.args.betas[0],
                beta2=self.args.betas[1],
                # momentum=self.args.momentum,
                margin=self.args.margin,
                weight_decay=self.args.weight_decay,
                epoch_decay=self.args.epoch_decay,
            )
            # self.lr_scheduler.add(
            #     StepSchedulerWithWarmup(optimizer, self.lr, self.args.lr_steps, self.args.warmup_epochs, key='lr0')
            # )
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    self.models.parameters(), 
                    self.lr,
                    weight_decay=self.args.weight_decay,
                    momentum=self.args.momentum
                )
            elif self.args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(
                    self.models.parameters(), 
                    self.lr,
                    weight_decay=self.args.weight_decay
                )
            else:
                raise NotImplementedError("Optimizer not implemented")

        ## for lr scheduler
        self.lr_scheduler.add(
            StepSchedulerWithWarmup(optimizer, self.lr, self.args.lr_steps, self.args.warmup_epochs)
        )

        # self.lr_scheduler = CosineSchedulerWithWarmup(optimizer, self.lr, self.args.warmup_epochs, self.args.epochs, self.lr / 1e2)

        return optimizer

    def train_dataloader(self):
        train_sampler = SubsetRandomSampler(random.sample(range(len(self.train_dataset)), len(self.train_dataset)))
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_sampler, num_workers=self.args.workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.args.workers)
        # return s


def main(args):
    # Define the CIFAR10 dataset
    # train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    ###########################
    # PARAMETERS
    ###########################
    if args.debug:
        args.workers = 0
    
    # seed for reproducibility
    if args.seed is not None:
        warnings.warn(f"You have seeded training with seed {args.seed}")
        set_seed(args.seed)
        pl.seed_everything(args.seed, workers=True)
    else:
        warnings.warn(f"You have not seeded training")

    if not torch.cuda.is_available():
        warnings.warn("No GPU available: training will be extremely slow")

    ###########################
    # DATASET
    ###########################

    info = INFO[args.dataset]
    task = info['task']
    # n_channels = 3 if args.as_rgb else info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    print('==> Preparing data...')

    train = DataClass('train', download=True, as_rgb=True)
    ndim = train.imgs.ndim
    # mean, std = (train.imgs / 255).mean().item(), (train.imgs / 255).std().item()
    mean,std = 0.5, 0.5
    if hasattr(augments, args.dataset):
        train_transform, eval_transform = getattr(augments, args.dataset)(ndim, args, mean, std)    
    else:
        train_transform, eval_transform = getattr(augments, args.augmentations)(ndim, args, mean, std)
    
    train_dataset = DataClass(split='train', transform=train_transform, download=True, as_rgb=True)
    val_dataset = DataClass(split='val', transform=eval_transform, download=True, as_rgb=True)
    test_dataset = DataClass(split='test', transform=eval_transform, download=True, as_rgb=True)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train and validate the bagged ensemble model
    num_models = 3
    ensemble_model = EnsembleModel(args,
                                   img_shape=train_dataset[0][0].shape, 
                                   num_outputs=train_dataset.labels.shape[1],
                                   num_models=num_models, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset
                                )   
    # trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=10)
    trainer = pl.Trainer(
        accelerator='gpu' if not args.debug else 'cpu',
        deterministic=True if args.seed is not None else False,
        max_epochs=args.epochs,
        check_val_every_n_epoch=int(args.evaluate_every),
        # logger=logger,
        log_every_n_steps=1,
        # callbacks=callbacks,

        fast_dev_run=args.debug,   # for testing training and validation
        num_sanity_val_steps=2,
    )
    trainer.fit(ensemble_model)
    print("Done Training")


if __name__ == '__main__':
    from arguments import args as a
    main(a)


# def main(args):
#     ###########################
#     # PARAMETERS
#     ###########################
#     if args.debug:
#         args.workers = 0
    
#     # seed for reproducibility
#     if args.seed is not None:
#         warnings.warn(f"You have seeded training with seed {args.seed}")
#         set_seed(args.seed)
#         pl.seed_everything(args.seed, workers=True)
#     else:
#         warnings.warn(f"You have not seeded training")

#     if not torch.cuda.is_available():
#         warnings.warn("No GPU available: training will be extremely slow")

#     ###########################
#     # DATASET
#     ###########################

#     info = INFO[args.dataset]
#     task = info['task']
#     # n_channels = 3 if args.as_rgb else info['n_channels']
#     n_classes = len(info['label'])
#     DataClass = getattr(medmnist, info['python_class'])

#     print('==> Preparing data...')

#     train = DataClass('train', download=True, as_rgb=True)
#     ndim = train.imgs.ndim
#     # mean, std = (train.imgs / 255).mean().item(), (train.imgs / 255).std().item()
#     mean,std = 0.5, 0.5
#     if hasattr(augments, args.dataset):
#         train_transform, eval_transform = getattr(augments, args.dataset)(ndim, args, mean, std)    
#     else:
#         train_transform, eval_transform = getattr(augments, args.augmentations)(ndim, args, mean, std)
    
#     train_dataset = DataClass(split='train', transform=train_transform, download=True, as_rgb=True)
#     val_dataset = DataClass(split='val', transform=eval_transform, download=True, as_rgb=True)
#     test_dataset = DataClass(split='test', transform=eval_transform, download=True, as_rgb=True)

#     # test_labels[test_labels != args.pos_class] = 999
#     # test_labels[test_labels == args.pos_class] = 1
#     # test_labels[test_labels == 999] = 0

#     print(f"==> Positive/negative samples: {(train_dataset.labels == 1).sum()}/{(train_dataset.labels == 0).sum()}=>{(train_dataset.labels == 1).sum()/train_dataset.labels.shape[0]}")

#     if args.sampler is not None:
#         sampler = DualSampler(
#             dataset=train_dataset, 
#             batch_size=args.batch_size, 
#             shuffle=True, 
#             sampling_rate=args.sampler,
#             #sampling_rate=(train_dataset.labels == 1).sum()/train_dataset.labels.shape[0]
#         )
#     else:
#         sampler = None
#     # train_dataloader = DataLoader(
#     #     dataset=train_dataset,
#     #     batch_size=args.batch_size,
#     #     sampler=sampler,
#     #     shuffle=sampler is None,
#     #     num_workers=args.workers, 
#     # )
#     val_dataloader = DataLoader(
#         dataset=val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.workers, 
#     )
#     test_dataloader = DataLoader(
#         dataset=test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.workers, 
#     )

#     ###########################
#     # Logger
#     ###########################

#     logger_base = os.path.join(args.save_dir, args.dataset, args.name)
#     if args.resume is None:
#         logger = pl.loggers.TensorBoardLogger(
#             save_dir=logger_base,
#             name=f"{args.augmentations}_{args.loss_type}_{args.batch_size}", 
#             # log_graph=True,
#         )
#     else:
#         logdir = args.resume.split('/')[1:-1]
#         logger = pl.loggers.TensorBoardLogger(
#             save_dir=logger_base,
#             name=os.path.join(*logdir[:-1]),
#             version=logdir[-1],
#             # log_graph=True,
#         )

#     ###########################
#     # Model
#     ###########################

#     n_outputs = train_dataset.labels.shape[1]
#     model_task = Module(
#         args=args,
#         img_shape=train_dataset[0][0].shape,
#         num_outputs=n_outputs,
#     )
#     if args.pretrained is not None:
#         # args.pretrained = os.path.join(logger_base, args.pretrained)
#         print(f"==>Loading pretrained model from {args.pretrained}")
#         model = Module.load_from_checkpoint(args.pretrained).model
        
#         if args.freeze:
#             for i, param in enumerate(model.parameters()):
#                 param.requires_grad = False

#         if 512 != n_outputs:
#             model.fc = nn.Linear(512, n_outputs, bias=True)
#         model_task.model = model

#     ###########################
#     # CALLBACKS
#     ###########################

#     callbacks = [
#         pl.callbacks.LearningRateMonitor(),
#         # pl.callbacks.DeviceStatsMonitor(),  # monitors and logs device stats, useful to find memory usage
#     ]

#     save_path = logger.log_dir

#     ## callback for saving checkpoints
#     checkpoint_cb_every = pl.callbacks.ModelCheckpoint(
#         dirpath=save_path, 
#         filename="last-{epoch:02d}-{val_auc:.4f}",
#         monitor="step",
#         mode="max",
#         save_top_k=1,
#         every_n_epochs=args.save_every_epochs,
#         save_on_train_epoch_end=True,
#         # train_time_interval=,
#         # every_n_train_steps=,
#         save_last=not args.use_best_model,
#     )
#     callbacks.append(checkpoint_cb_every)

#     if args.loss_type != 'pre':
#         checkpoint_cb_bestk = pl.callbacks.ModelCheckpoint(
#             dirpath=save_path, 
#             filename="best_auc-{epoch:02d}-{val_auc:.4f}",
#             save_top_k=1, 
#             monitor='val_auc',
#             mode='max',
#             verbose=True,
#             # save_on_train_epoch_end=False,
#             save_last=args.use_best_model,
#         )
#         callbacks.append(checkpoint_cb_bestk)

#         # early stopping
#         if args.early_stopping_patience is not None:
#             early_stopping = pl.callbacks.EarlyStopping(
#                 monitor='val_auc',
#                 mode='max',
#                 patience=args.early_stopping_patience,
#                 verbose=True,
#             )
#             callbacks.append(early_stopping)

#     ###########################
#     # TRAINER
#     ###########################

#     # may increase performance but lead to unstable training
#     # torch.set_float32_matmul_precision("high")
#     trainer = pl.Trainer(
#         accelerator='gpu' if not args.debug else 'cpu',
#         deterministic=True if args.seed is not None else False,
#         # precision="16-mixed",   # reduce memory, can improve performance but might lead to unstable training
        
#         max_epochs=args.epochs,
#         # max_time="00:1:00:00",
#         # max_steps=,

#         check_val_every_n_epoch=int(args.evaluate_every),
#         # val_check_interval=args.evaluate_every,
#         logger=logger,
#         log_every_n_steps=1,
#         callbacks=callbacks,

#         fast_dev_run=args.debug,   # for testing training and validation
#         num_sanity_val_steps=2,
#         # limit_train_batches=1.0 if not args.debug else 0.01,  # to test what happens after an epoch
#         # overfit_batches=0.01,

#         # profiler='pytorch',    # advanced profiling to check for bottlenecks
#     )

#     ###########################
#     # RUN MODEL
#     ###########################

#     # ## call tune to find lr and batch size
#     # from lightning.pytorch.tuner import Tuner
#     # tuner = pl.tuner.Tuner(trainer)
#     # lr_finder = tuner.lr_find(model_task, train_dataloaders=train_dataloader)
#     # print(lr_finder.results)
#     # fig = lr_finder.plot(suggest=True)
#     # fig.show()
#     # # new_lr = lr_finder.suggestion()
#     # # batch_size = tuner.scale_batch_size(model_task, train_dataloaders=train_dataloader)
#     # return

#     # fit the model
#     if args.test is None:
#         print("Fitting model...")
#         trainer.fit(
#             model=model_task,
#             train_dataloaders=train_dataloader,
#             val_dataloaders=val_dataloader,
#             ckpt_path=args.resume,
#         )

#     if args.loss_type == 'pre':
#         return


#     if args.test is not None:
#         test_models = [('test', args.test)]
#     else:
#         test_models = [('last', checkpoint_cb_every.best_model_path), ('best', checkpoint_cb_bestk.best_model_path)]        
#     for name, cp in test_models:
#         # cp = checkpoint_cb_bestk if args.use_best_model else checkpoint_cb_every

#         ## validate model
#         results_val = trainer.validate(
#             model=model_task,
#             dataloaders=val_dataloader, 
#             ckpt_path=cp,
#             verbose=True,
#         )

#         ## test model
#         results_test = trainer.test(
#             model=model_task,
#             dataloaders=test_dataloader, 
#             ckpt_path=cp,
#             verbose=True,
#         )

#         logger.experiment.add_scalar(f"final_val_auc_{name}", results_val[0]['val_auc'], global_step=trainer.global_step+100)
#         logger.experiment.add_scalar(f"final_test_auc_{name}", results_test[0]['test_auc'], global_step=trainer.global_step+100)

#         results = {}
#         results['val'] = results_val[0]
#         results['test'] = results_test[0]
#         results['args'] = args
#         results['dataset'] = args.dataset
#         results['name'] = logger.name

#         save_pickle(results, os.path.join(save_path, f"results_{logger.version}_{name}.pkl"))
#         save_dict(results, os.path.join(save_path, f"results_{logger.version}_{name}.dict"), as_str=True)


# if __name__ == '__main__':
#     from arguments import args as a
#     main(a)
