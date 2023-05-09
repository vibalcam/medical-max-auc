import argparse
from functools import partial
import gc
import math
import os
import shutil
from typing import Any, Optional
import warnings
from matplotlib import pyplot as plt
from models.pretraining import similarity_loss
from models.schedulers import CosineSchedulerWithWarmup, SchedulerCollection, StepSchedulerWithWarmup
from models.utils import ClassificationMetrics, get_logger_name, model_to_syncbn, save_dict, save_pickle, set_seed
from pathlib import Path

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from libauc.losses import AUCMLoss, CompositionalAUCLoss, AUCM_MultiLabel, AUCM_MultiLabel_V2
from libauc import optimizers
import medmnist 
from models import augments
from medmnist import INFO
from models.sampler import DualSampler
from torchsampler import ImbalancedDatasetSampler


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

        self.lr_scheduler = None
        # get metrics and model
        self.define_metrics()
        if self.args.model_per_task:
            self.model = nn.ModuleList([self.create_model(1) for _ in range(num_outputs)])
        else:
            self.model = self.create_model(num_outputs)

        # define loss function
        if self.args.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss_type == 'auc':
            if num_outputs == 1:
                self.criterion = AUCMLoss()
            else:
                if self.args.model_per_task:
                    self.criterion = [AUCMLoss() for _ in range(num_outputs)]
                else:
                    self.criterion = AUCM_MultiLabel(num_classes=num_outputs)
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

        # params = dict(task='binary' if self.hparams.num_outputs==1 else 'multilabel', num_labels=self.hparams.num_outputs, average='macro')
        # self.train_metrics = torchmetrics.MetricCollection({
        #     'aucroc': torchmetrics.AUROC(**params),
        #     'acc': torchmetrics.Accuracy(**params),
        # }, prefix='train_')
        # self.val_metrics = self.train_metrics.clone(prefix='val_')
        # self.test_metrics = self.train_metrics.clone(prefix='test_')

    def create_model(self, n_outputs):
        # if self.args.loss_type=='pre':
        #     n_outputs = 256

        from libauc.models import resnet18 as ResNet18
        model = ResNet18(pretrained=False, num_classes=n_outputs)
        dropout = nn.Dropout2d
        
        if len(self.hparams.img_shape) == 4:
            if self.args.type_3d == '3d':
                ## use 3d conv
                ## https://paperswithcode.com/lib/torchvision/resnet-3d#:~:text=ResNet%203D%20is%20a%20type,convolutions%20in%20the%20top%20layers
                import torchvision.models as tm
                model = tm.video.r3d_18(pretrained=False, num_classes=n_outputs)
                dropout = nn.Dropout3d
            elif self.args.type_3d == 'channels':
                ## consider 3rd dimension as channel
                model.conv1 = torch.nn.Conv2d(self.hparams.img_shape[-1], 64, kernel_size=7, stride=2, padding=3, bias=False)
            else:
                raise NotImplementedError()

            # # use 3d converter
            # from acsconv.converters import Conv3dConverter
            # model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
            # model.cuda()
        
        if self.args.dropout is not None:
            def append_dropout(m, rate):
                for name, module in m.named_children():
                    if len(list(module.children())) > 0:
                        append_dropout(module, rate)
                    if isinstance(module, nn.ReLU):
                        new = nn.Sequential(module, dropout(p=rate))
                        setattr(m, name, new)
            append_dropout(model, self.args.dropout)

        if self.args.loss_type=='pre':
            model.fc = torch.nn.Identity()

        return model

    def configure_optimizers(self):
        self.lr_scheduler = SchedulerCollection()
        optims = []
        n_optims = self.hparams.num_outputs if self.args.model_per_task else 1

        for k in range(n_optims):
            if self.args.model_per_task:
                m = self.model[k]
                l = self.criterion[k]
            else:
                m = self.model
                l = self.criterion
            ## optimizer
            if self.args.loss_type == 'auc':
                optimizer = optimizers.PESG(
                    m, 
                    loss_fn=l, 
                    lr=self.lr, 
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay,
                    epoch_decay=self.args.epoch_decay,
                    margin=self.args.margin,
                )
            elif self.args.loss_type == 'comp':
                optimizer = optimizers.PDSCA(
                    m, 
                    loss_fn=l, 
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
                        m.parameters(), 
                        self.lr,
                        weight_decay=self.args.weight_decay,
                        momentum=self.args.momentum
                    )
                elif self.args.optimizer == 'adamw':
                    optimizer = torch.optim.AdamW(
                        m.parameters(), 
                        self.lr,
                        weight_decay=self.args.weight_decay
                    )
                else:
                    raise NotImplementedError("Optimizer not implemented")

            optims.append(optimizer)
            ## for lr scheduler
            self.lr_scheduler.add(
                StepSchedulerWithWarmup(optimizer, self.lr, self.args.lr_steps, self.args.warmup_epochs)
            )

        # self.lr_scheduler = CosineSchedulerWithWarmup(optimizer, self.lr, self.args.warmup_epochs, self.args.epochs, self.lr / 1e2)

        if len(optims) == 1:
            return optims[0]

        return optimizer
    
    def forward(self, x):
        if len(self.hparams.img_shape) == 4 and self.args.type_3d == 'channels':
            x = x[:,0,...]
        if self.args.model_per_task:
            return torch.cat([m(x) for m in self.model], 1)
        else:
            return self.model(x)

    def on_train_epoch_start(self):
        # adjust_learning_rate(self.optimizers(), self.lr, self.current_epoch, self.args)
        self.lr_scheduler.step(self.current_epoch)
        self.model.train()

    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        target = target.float()
        if self.args.loss_type=='auc':
            if self.args.model_per_task:
                pred = torch.sigmoid(output)
                loss = [self.criterion[k](pred[:,k], target[:,k]) for k in range(output.shape[1])]
                loss = torch.cat(loss).mean()
            else:
                loss = self.criterion(torch.sigmoid(output), target)
        elif self.args.loss_type=='pre':
            loss = self.criterion(
                output, 
                target,
                T=0.1, 
                device=self.device,
            )
        else:
            loss = self.criterion(output, target)

        if output.isnan().any():
            warnings.warn("Nan values being generated")
        if loss.isnan():
            warnings.warn("Getting nan loss")

        # calculate metrics
        if self.args.loss_type != 'pre':
            y_pred = torch.sigmoid(output)
            r = self.train_metrics.update(target.int(), y_pred)
            self.log_dict(r)

            # self.train_metrics(y_pred, target.int().detach())
            # self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # log loss for training step and average loss for epoch
        self.log_dict({
            "train_loss": loss,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
    
    def on_train_epoch_end(self) -> None:
        if self.args.loss_type == 'pre':
            return
        
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):   
        if self.args.loss_type == 'pre':
            return

        # run through model
        x, target = batch[:2]
        output = self(x)

        if output.isnan().any():
            warnings.warn("Nan values being generated")
            return

        # calculate metrics
        self.val_metrics.update(target.int(), torch.sigmoid(output))
        # y_pred = torch.sigmoid(output).float().detach()
        # target = target.int().detach()
        # self.val_metrics.update(y_pred, target)
        # self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.lr_scheduler is not None:
            self.lr_scheduler.update(None)

    def on_validation_epoch_end(self) -> None:
        if self.args.loss_type == 'pre' or len(self.val_metrics) == 0:
            return
        
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.model.eval()

    def test_step(self, batch, batch_idx):
        # run through model
        images, target = batch
        output = self(images)

        self.test_metrics.update(target.int(), torch.sigmoid(output))

        # y_pred = torch.sigmoid(output).float().detach()
        # target = target.int().detach()
        # self.test_metrics.update(y_pred, target)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        # self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)


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
    if args.train_on_val is not None:
    #     warnings.warn(f"You are training on the train set and validation set")
        # print(type(train_dataset))
        val_for_train_dataset = DataClass(split='val', transform=train_transform, download=True, as_rgb=True)
    #     train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    # test_labels[test_labels != args.pos_class] = 999
    # test_labels[test_labels == args.pos_class] = 1
    # test_labels[test_labels == 999] = 0

    print(f"==> Positive/negative samples: {(train_dataset.labels == 1).sum()}/{(train_dataset.labels == 0).sum()}=>{(train_dataset.labels == 1).sum()/train_dataset.labels.shape[0]}")

    if args.oversample is not None:
        sampler=ImbalancedDatasetSampler(train_dataset)
    elif args.sampler is not None:
        sampler = DualSampler(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            sampling_rate=args.sampler,
            # sampling_rate=(train_dataset.labels == 1).sum()/train_dataset.labels.shape[0]
        ) 
    else:
        sampler = None

    if args.train_on_val is not None:
        warnings.warn("You are training on the train set and validation set")
        train_dataloader = DataLoader(
            dataset=torch.utils.data.ConcatDataset([train_dataset, val_for_train_dataset]),
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=args.workers, 
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
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

    # logger_base = os.path.join(args.save_dir, args.dataset, args.name)
    logger_base = os.path.join(args.save_dir, args.dataset)
    if args.resume is None:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=logger_base,
            name=args.name, 
            # name=f"{args.augmentations}_{args.loss_type}_{args.batch_size}", 
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

    n_outputs = train_dataset.labels.shape[1]
    model_task = Module(
        args=args,
        img_shape=train_dataset[0][0].shape,
        num_outputs=n_outputs,
    )
    if args.pretrained is not None and args.test is None:
        # args.pretrained = os.path.join(logger_base, args.pretrained)
        print(f"==>Loading pretrained model from {args.pretrained}")
        model = Module.load_from_checkpoint(args.pretrained).model
        
        if args.freeze:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False

        if 512 != n_outputs:
            model.fc = nn.Linear(512, n_outputs, bias=True)
        model_task.model = model

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
        filename="last-{epoch:02d}-{val_auc:.4f}",
        monitor="step",
        mode="max",
        save_top_k=1,
        every_n_epochs=args.save_every_epochs,
        save_on_train_epoch_end=True,
        # train_time_interval=,
        # every_n_train_steps=,
        # save_last=not args.use_best_model,
    )
    callbacks.append(checkpoint_cb_every)

    if args.loss_type != 'pre':
        checkpoint_cb_bestk = pl.callbacks.ModelCheckpoint(
            dirpath=save_path, 
            filename="best-{epoch:02d}-{val_auc:.4f}",
            save_top_k=1, 
            monitor='val_auc',
            mode='max',
            verbose=True,
            # save_on_train_epoch_end=False,
            # save_last=args.use_best_model,
        )
        callbacks.append(checkpoint_cb_bestk)

        # early stopping
        if args.early_stopping_patience is not None:
            early_stopping = pl.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=args.early_stopping_patience,
                verbose=True,
            )
            callbacks.append(early_stopping)

    ###########################
    # TRAINER
    ###########################

    # may increase performance but lead to unstable training
    if args.use_16:
        torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        accelerator='gpu' if not args.debug else 'cpu',
        deterministic=True if args.seed is not None else False,
        precision="16-mixed" if args.use_16 else '32-true',   # reduce memory and faster training, but might lead to unstable training
        
        max_epochs=args.epochs,
        # max_time="00:1:00:00",
        # max_steps=,

        check_val_every_n_epoch=int(args.evaluate_every),
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

    ## fit the model
    if args.test is None:
        print("Fitting model...")
        trainer.fit(
            model=model_task,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.resume,
        )

    if args.loss_type == 'pre':
        return

    if args.test is not None:
        test_models = [('test', args.test)]
    else:
        test_models = [('last', checkpoint_cb_every.best_model_path), ('best', checkpoint_cb_bestk.best_model_path)]   

        if args.use_best_model:
            ck = checkpoint_cb_bestk.best_model_path
            warnings.warn("==>Using best model")
        else:
            ck = checkpoint_cb_every.best_model_path
            warnings.warn("==>Using last model")

        p = os.path.join(Path(ck).parent, 'best.ckpt')
        shutil.copy(ck, p)

    for name, cp in test_models:
        # cp = checkpoint_cb_bestk if args.use_best_model else checkpoint_cb_every

        ## validate model
        results_val = trainer.validate(
            model=model_task,
            dataloaders=val_dataloader, 
            ckpt_path=cp,
            verbose=False,
        )

        ## test model
        results_test = trainer.test(
            model=model_task,
            dataloaders=test_dataloader, 
            ckpt_path=cp,
            verbose=False,
        )

        logger.experiment.add_scalar(f"final_val_auc_{name}", results_val[0]['val_auc'], global_step=trainer.global_step+100)
        logger.experiment.add_scalar(f"final_test_auc_{name}", results_test[0]['test_auc'], global_step=trainer.global_step+100)

        results = {}
        results['val'] = results_val[0]
        results['test'] = results_test[0]
        results['args'] = args
        results['dataset'] = args.dataset
        results['name'] = logger.name

        save_pickle(results, os.path.join(save_path, f"results_{logger.version}_{name}.pkl"))
        save_dict(results, os.path.join(save_path, f"results_{logger.version}_{name}.dict"), as_str=True)

    return results


if __name__ == '__main__':
    from arguments import args as a
    main(a)
