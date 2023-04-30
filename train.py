import argparse
import gc
import math
import os
from typing import Any, Optional
import warnings
from matplotlib import pyplot as plt
from models.schedulers import CosineSchedulerWithWarmup, StepSchedulerWithWarmup
from models.utils import model_to_syncbn, save_pickle, set_seed

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from libauc.losses.auc import tpAUC_KL_Loss
from libauc.optimizers import SOTAs
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
import medmnist 
from models import augments
from medmnist import INFO, Evaluator


parser = argparse.ArgumentParser(description='')

# general options
parser.add_argument('--name', default='default', type=str, help='name of experiment')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)

parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=44444, type=int, help='seed for initializing training. ')
parser.add_argument('--debug', action='store_true', help='To debug code')

# optimizer options
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-wd', '--weight_decay', default=1e-2, type=float, metavar='W', help='weight decay (default: 1e-6)', dest='weight_decay')
parser.add_argument('--loss_type', default='bce', type=str, help='loss type of pretrained (default: bce)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial (base) learning rate')
parser.add_argument('--optimizer', default='adamw', type=str, choices=['sgd', 'adamw'], help='optimizer used (default: sgd)')
parser.add_argument('--warmup_epochs', default=0, type=float, help='number of warmup epochs')
parser.add_argument('--lr_steps', default=[50, 75], type=int, nargs="+", help='epochs to decay learning rate by 10')

# dataset 
parser.add_argument('--save_dir', default='./saved_models/', type=str)
parser.add_argument('--results_file', default='results', type=str)
parser.add_argument('--dataset', type=str, default="breastmnist", choices=["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",])
parser.add_argument('--augmentations', type=str, default="basic")
parser.add_argument('--aug_args', type=str, default='gn.ra')

# saving
parser.add_argument('--save_every_epochs', default=5, type=int, help='number of epochs to save checkpoint')
parser.add_argument('-e', '--evaluate_every', default=5, type=float, help='evaluate model on validation set every # epochs')
parser.add_argument('--early_stopping_patience', default=None, type=int, help='patience for early stopping')
parser.add_argument('--use_best_model', action='store_true', help='use best model for evaluation')

# other model
parser.add_argument('--dropout', type=float, default=None, help='dropout rate')
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--pretrain_type", type=str, default='bce')
parser.add_argument("--type_3d", type=str, default='3d')

args = parser.parse_args()


class Module(pl.LightningModule):
    def __init__(
        self,
        args,
        img_shape,
        num_outputs,
        **kwargs,
    ):
        super().__init__()

        # save hyperparameters as attribute
        self.save_hyperparameters(ignore=['model'])
        self.args = args

        # define batchsize and learning rate
        self.batch_size = self.args.batch_size
        ## infer learning rate
        self.lr = self.args.lr
        # self.init_lr = self.args.lr * self.batch_size / 256
        print('initial learning rate:', self.lr)

        # get metrics and model
        self.define_metrics()
        self.model = self.create_model(self.hparams.num_outputs)

        # define loss function
        if self.args.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        # elif args.loss_type == 'auc':
        #     self.criterion = AUCMLoss()
        else:
            raise NotImplementedError()
    
    def define_metrics(self):
        params = dict(task='binary' if self.hparams.num_outputs==1 else 'multilabel', num_labels=self.hparams.num_outputs, average='macro')
        self.train_metrics = torchmetrics.MetricCollection({
            'aucroc': torchmetrics.AUROC(**params),
            'acc': torchmetrics.Accuracy(**params),
        }, prefix='train_')
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')

    def create_model(self, n_outputs):
        # from models.resnet import resnet18 as ResNet18
        from libauc.models import resnet18 as ResNet18
        model = ResNet18(pretrained=False, num_classes=n_outputs)
        
        if len(self.hparams.img_shape) == 4:
            if self.args.type_3d == '3d':
                ## use 3d conv
                ## https://paperswithcode.com/lib/torchvision/resnet-3d#:~:text=ResNet%203D%20is%20a%20type,convolutions%20in%20the%20top%20layers
                import torchvision.models as tm
                model = tm.video.r3d_18(pretrained=False, num_classes=n_outputs)
            elif self.args.type_3d == 'channels':
                ## consider 3rd dimension as channel
                model.conv1 = torch.nn.Conv2d(self.hparams.img_shape[-1], 64, kernel_size=7, stride=2, padding=3, bias=False)
            else:
                raise NotImplementedError()

            # # use 3d converter
            # from acsconv.converters import Conv3dConverter
            # model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
            # model.cuda()
        
        # todo! add dropout for 3d
        if self.args.dropout is not None:
            def append_dropout(m, rate):
                for name, module in m.named_children():
                    if len(list(module.children())) > 0:
                        append_dropout(module, rate)
                    if isinstance(module, nn.ReLU):
                        new = nn.Sequential(module, nn.Dropout2d(p=rate))
                        setattr(m, name, new)
            append_dropout(model, self.args.dropout)

        return model

    def configure_optimizers(self):
        ## optimizer
        if self.args.loss_type == 'auc':
            raise NotImplementedError()
            # optimizer = PESG(
            #     self.model, 
            #     loss_fn=self.criterion, 
            #     lr=self.lr, 
            #     momentum=self.args.momentum,
            #     weight_decay=self.args.weight_decay,
            # )
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    self.model.parameters(), 
                    self.lr,
                    weight_decay=self.args.weight_decay,
                    momentum=self.args.momentum
                )
            elif self.args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    self.lr,
                    weight_decay=self.args.weight_decay
                )
            else:
                raise NotImplementedError("Optimizer not implemented")

        ## for lr scheduler
        self.lr_scheduler = StepSchedulerWithWarmup(optimizer, self.lr, self.args.lr_steps, self.args.warmup_epochs)
        # self.lr_scheduler = CosineSchedulerWithWarmup(optimizer, self.lr, self.args.warmup_epochs, self.args.epochs, self.lr / 1e2)

        return optimizer
    
    def forward(self, x):
        if len(self.hparams.img_shape) == 4 and self.args.type_3d == 'channels':
            x = x[:,0,...]
        return self.model(x)

    def on_train_epoch_start(self):
        # adjust_learning_rate(self.optimizers(), self.lr, self.current_epoch, self.args)
        self.lr_scheduler.step(self.current_epoch)
        self.model.train()

    def training_step(self, batch, batch_idx):
        if self.args.loss_type=='auc':
            # loss = self.criterion(torch.sigmoid(output).float(), target, index.long())
            # loss = self.criterion(torch.sigmoid(output).float(), target, index[:self.pos_samples].long())
            raise NotImplementedError()
        else:
            x, target = batch
            output = self(x)
            loss = self.criterion(output, target.float())

        if output.isnan().any():
            warnings.warn("Nan values being generated")
        if loss.isnan():
            warnings.warn("Getting nan loss")

        # calculate metrics
        y_pred = torch.sigmoid(output).detach().float()
        self.train_metrics(y_pred, target.int().detach())

        # log loss for training step and average loss for epoch
        self.log_dict({
            "train_loss": loss,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        ## compute gradient and do SGD step
        ## automatically done by lightning
        ## can be disabled (https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html)
        # optimizer.zero_grad()
        # outputs = model(input)
        # loss = loss_f(output, labels)
        # loss.backward()
        # optimizer.step()

        # torch.cuda.empty_cache()
        # gc.collect()
        
        return loss

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):   
        # run through model
        x, target = batch[:2]
        output = self(x)

        # calculate metrics
        y_pred = torch.sigmoid(output).float().detach()
        target = target.int().detach()
        self.val_metrics.update(y_pred, target)

        self.lr_scheduler.update(None)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_start(self) -> None:
        self.model.eval()

    def test_step(self, batch, batch_idx):
        # run through model
        images, target = batch
        output = self(images)

        y_pred = torch.sigmoid(output).float().detach()
        target = target.int().detach()
        self.test_metrics.update(y_pred, target)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)


# class dataset(torch.utils.data.Dataset):
#     def __init__(self, inputs, targets, trans=None):
#         self.x = inputs
#         self.y = targets
#         self.trans=trans

#     def __len__(self):
#         return self.x.size()[0]

#     def __getitem__(self, idx):
#         if self.trans == None:
#             return (self.x[idx], self.y[idx], idx)
#         else:
#             return (self.trans(self.x[idx]), self.y[idx], idx) 


def main(args):
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

    # test_labels[test_labels != args.pos_class] = 999
    # test_labels[test_labels == args.pos_class] = 1
    # test_labels[test_labels == 999] = 0

    print(f"==> Positive/negative samples: {(train_dataset.labels == 1).sum()}/{(train_dataset.labels == 0).sum()}")

    # sampler = DualSampler(train_dataset, args.batch_size, sampling_rate=0)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers, 
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, 
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, 
    )

    ###########################
    # Logger
    ###########################

    logger_base = os.path.join(args.save_dir, args.dataset, args.name)
    if args.resume is None:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=logger_base,
            name=os.path.join(args.augmentations, args.loss_type), 
            # log_graph=True,
        )
    else:
        logdir = args.resume.split('/')[1:-1]
        logger = pl.loggers.TensorBoardLogger(
            save_dir=logger_base,
            name=os.path.join(*logdir[:-1]),
            version=logdir[-1],
            # log_graph=True,
        )

    ###########################
    # Model
    ###########################

    if args.pretrained is not None:
        args.pretrained = os.path.join(logger_base, args.pretrained)
        print(f"==>Loading pretrained model from {args.pretrained}")
        model_task = Module.load_from_checkpoint(
            args.pretrained,
            args=args,
            img_shape=train_dataset[0][0].shape,
            num_outputs=train_dataset.labels.shape[1],
        )
    else:
        model_task = Module(
            args=args,
            img_shape=train_dataset[0][0].shape,
            num_outputs=train_dataset.labels.shape[1],
        )

    ###########################
    # CALLBACKS
    ###########################

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        # pl.callbacks.DeviceStatsMonitor(),  # monitors and logs device stats, useful to find memory usage
    ]

    save_path = logger.log_dir

    ## callback for saving checkpoints
    checkpoint_cb_every = pl.callbacks.ModelCheckpoint(
        dirpath=save_path, 
        filename="last-{epoch:02d}-{val_aucroc:.4f}",
        monitor="step",
        mode="max",
        save_top_k=1,
        every_n_epochs=args.save_every_epochs,
        save_on_train_epoch_end=True,
        # train_time_interval=,
        # every_n_train_steps=,
        # save_last=False,
    )
    callbacks.append(checkpoint_cb_every)

    checkpoint_cb_bestk = pl.callbacks.ModelCheckpoint(
        dirpath=save_path, 
        filename="best_auc-{epoch:02d}-{val_aucroc:.4f}",
        save_top_k=1, 
        monitor='val_aucroc',
        mode='max',
        verbose=True,
        # save_on_train_epoch_end=False,
        # save_last=False,
    )
    callbacks.append(checkpoint_cb_bestk)

    # early stopping
    if args.early_stopping_patience is not None:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor='val_aucroc',
            mode='max',
            patience=args.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping)

    ###########################
    # TRAINER
    ###########################

    # may increase performance but lead to unstable training
    # torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        accelerator='gpu' if not args.debug else 'cpu',
        deterministic="warn" if args.seed is not None else False,
        # precision="16-mixed",   # reduce memory, can improve performance but might lead to unstable training
        
        max_epochs=args.epochs,
        # max_time="00:1:00:00",
        # max_steps=,

        check_val_every_n_epoch=args.evaluate_every,
        # val_check_interval=args.evaluate_every,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,

        fast_dev_run=args.debug,   # for testing training and validation
        num_sanity_val_steps=2,
        # limit_train_batches=1.0 if not args.debug else 0.01,  # to test what happens after an epoch
        # overfit_batches=0.01,

        # profiler='pytorch',    # advanced profiling to check for bottlenecks
    )

    ###########################
    # RUN MODEL
    ###########################

    # ## call tune to find lr and batch size
    # from lightning.pytorch.tuner import Tuner
    # tuner = pl.tuner.Tuner(trainer)
    # lr_finder = tuner.lr_find(model_task, train_dataloaders=train_dataloader)
    # print(lr_finder.results)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # # new_lr = lr_finder.suggestion()
    # # batch_size = tuner.scale_batch_size(model_task, train_dataloaders=train_dataloader)
    # return

    # fit the model
    print("Fitting model...")
    trainer.fit(
        model=model_task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume,
    )

    ## validate model
    results_val = trainer.validate(
        model=model_task,
        dataloaders=val_dataloader, 
        ckpt_path=checkpoint_cb_every.best_model_path,
        # ckpt_path=checkpoint_cb_bestk.best_model_path,
        verbose=True,
    )

    ## test model
    cp = checkpoint_cb_bestk if args.use_best_model else checkpoint_cb_every
    results_test = trainer.test(
        model=model_task,
        dataloaders=test_dataloader, 
        ckpt_path=cp.best_model_path,
        # ckpt_path=checkpoint_cb_every.best_model_path,
        # ckpt_path=checkpoint_cb_bestk.best_model_path,
        verbose=True,
    )

    results = {}
    results['val'] = results_val[0]
    results['test'] = results_test[0]
    results['args'] = args
    results['dataset'] = args.dataset
    results['name'] = logger.name
    save_pickle(results, os.path.join(save_path, 'results.pkl'))


if __name__ == '__main__':
    main(args)
