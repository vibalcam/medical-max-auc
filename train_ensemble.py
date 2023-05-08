# import random
# from lightning.pytorch.utilities.types import EVAL_DATALOADERS
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, RandomSampler
# from torchvision.models import resnet18
# import pytorch_lightning as pl
# from train import Module as GeneralModel

# import argparse
# from functools import partial
# import gc
# import math
import os
# from typing import Any, Optional
import warnings
# from matplotlib import pyplot as plt
from models.pretraining import similarity_loss
# from models.schedulers import CosineSchedulerWithWarmup, SchedulerCollection, StepSchedulerWithWarmup
# from models.utils import ClassificationMetrics, get_logger_name, model_to_syncbn, save_dict, save_pickle, set_seed
from models.utils import ClassificationMetrics, save_dict, save_pickle, set_seed

# import numpy as np
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
import numpy as np
# from models.sampler import DualSampler
from train import Module as GeneralModel
from torchsampler import ImbalancedDatasetSampler


class EnsembleModel(pl.LightningModule):
    def __init__(
        self,
        args,
        img_shape,
        num_outputs,
        num_models,
        train_dataset,
        val_dataset,
        test_dataset,
        **kwargs,
    ):
        super(EnsembleModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.args = args
        self.num_models = num_models
        self.define_metrics()
        self.lr = self.args.lr
        self.models = nn.ModuleList([GeneralModel(args=args, img_shape=img_shape, num_outputs=num_outputs) for _ in range(3)])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.automatic_optimization = False

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

    def configure_optimizers(self):
        # ONLY WORKS FOR auc loss, to work with others need
        # self.lr_scheduler = SchedulerCollection()
        ## optimizer
        if self.args.loss_type == 'auc':
            optimizer0 = optimizers.PESG(
                self.models[0], 
                loss_fn=self.criterion, 
                lr=self.lr, 
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                epoch_decay=self.args.epoch_decay,
                margin=self.args.margin,
            )
            optimizer1 = optimizers.PESG(
                self.models[1], 
                loss_fn=self.criterion, 
                lr=self.lr, 
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                epoch_decay=self.args.epoch_decay,
                margin=self.args.margin,
            )
            optimizer2 = optimizers.PESG(
                self.models[2], 
                loss_fn=self.criterion, 
                lr=self.lr, 
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                epoch_decay=self.args.epoch_decay,
                margin=self.args.margin,
            )
        else:
            raise NotImplementedError("Optimizer not implemented")

        # self.lr_scheduler.add(StepSchedulerWithWarmup(optimizer, self.lr, self.args.lr_steps, self.args.warmup_epochs))
        return [optimizer0, optimizer1, optimizer2], []

    def forward(self, x):
        output = 0
        for model in self.models:
            output += model(x)
        return output / self.num_models

    def training_step(self, batch, batch_idx):
        opt0, opt1, opt2 = self.optimizers()
        x0, target0 = batch["d0"]
        x1, target1 = batch["d1"]
        x2, target2 = batch["d2"]

        output0 = self.models[0](x0)
        output1 = self.models[1](x1)
        output2 = self.models[2](x2)

        if self.args.loss_type=='auc':
            loss0 = self.criterion(torch.sigmoid(output0), target0.float())
            loss1 = self.criterion(torch.sigmoid(output1), target1.float())
            loss2 = self.criterion(torch.sigmoid(output2), target2.float())
            opt0.zero_grad()
            self.manual_backward(loss0)
            opt0.step()
            opt1.zero_grad()
            self.manual_backward(loss1)
            opt1.step()
            opt2.zero_grad()
            self.manual_backward(loss2)
            opt2.step()

            tensorboard_logs = {"train0_loss": loss0, "train1_loss": loss1, "train2_loss": loss2}
            self.log_dict(tensorboard_logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # elif self.args.loss_type=='pre':
        #     loss = self.criterion(output, target, T=0.1, device=self.device)
        # else:
        #     loss = self.criterion(output, target.float())
        # if output.isnan().any():
        #     warnings.warn("Nan values being generated")
        # if loss.isnan():
        #     warnings.warn("Getting nan loss")
        if self.args.loss_type != 'pre':
            ensemble_input = torch.concat([x0, x1, x2])
            ensemble_output = self(ensemble_input)
            concat_targets = torch.concat([target0, target1, target2])
            y_pred = torch.sigmoid(ensemble_output)
            r = self.train_metrics.update(concat_targets.int(), y_pred)
            self.log_dict(r)
        # tensorboard_logs = {"train0_loss": loss0, "train1_loss": loss1, "train2_loss": loss2}
        # self.log_dict(tensorboard_logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return {"loss": loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # run through model
        x, target = batch
        output = self(x)
        self.test_metrics.update(target.int(), torch.sigmoid(output))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):   
        if self.args.loss_type == 'pre':
            return
        # run through model
        x, target = batch[:2]
        output = self(x)
        # calculate metrics
        self.val_metrics.update(target.int(), torch.sigmoid(output))
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.update(None)

    def on_train_epoch_start(self):
        # self.lr_scheduler.step(self.current_epoch)
        self.models.train()

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.models.eval()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.models.eval()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        self.test_metrics.update(target.int(), torch.sigmoid(output))

    def train_dataloader(self):
        # train_size = len(self.train_dataset)
        if self.args.train_on_val is not None:
            concat_dataset=torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset]),
            train_sets = torch.utils.data.random_split(concat_dataset, [0.33, 0.33, 0.34])
        else:
            train_sets = torch.utils.data.random_split(self.train_dataset, [0.33, 0.33, 0.34])
        loaders = {}
        for i in range(len(train_sets)):
            if self.args.oversample is not None:
                train_set_indices = train_sets[i].indices
                subset_labels = np.squeeze(self.train_dataset.labels[train_set_indices])
                loaders["d"+str(i)] = DataLoader(train_sets[i],
                                                 batch_size=self.args.batch_size,
                                                 sampler=ImbalancedDatasetSampler(train_sets[i], labels=subset_labels),
                                                 num_workers=self.args.workers)
            else:
                loaders["d"+str(i)] = DataLoader(train_sets[i],
                                                 batch_size=self.args.batch_size, 
                                                 num_workers=self.args.workers)
        return loaders
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.workers)


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
    DataClass = getattr(medmnist, info['python_class'])

    print('==> Preparing data...')

    train = DataClass('train', download=True, as_rgb=True)
    ndim = train.imgs.ndim
    mean,std = 0.5, 0.5
    if hasattr(augments, args.dataset):
        train_transform, eval_transform = getattr(augments, args.dataset)(ndim, args, mean, std)    
    else:
        train_transform, eval_transform = getattr(augments, args.augmentations)(ndim, args, mean, std)
    
    train_dataset = DataClass(split='train', transform=train_transform, download=True, as_rgb=True)
    val_dataset = DataClass(split='val', transform=eval_transform, download=True, as_rgb=True)
    test_dataset = DataClass(split='test', transform=eval_transform, download=True, as_rgb=True)
    # if args.train_on_val:
    #     val_for_train_dataset = DataClass(split='val', transform=train_transform, download=True, as_rgb=True)

    ensemble_model = EnsembleModel(args,
                                   img_shape=train_dataset[0][0].shape, 
                                   num_outputs=train_dataset.labels.shape[1],
                                   num_models=args.num_models, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   test_dataset=test_dataset
                                ) 

    ###########################
    # LOGGER
    ###########################
    logger_base = os.path.join(args.save_dir, args.dataset, args.name)
    if args.resume is None:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=logger_base,
            name=f"{args.augmentations}_{args.loss_type}_{args.batch_size}", 
        )
    else:
        logdir = args.resume.split('/')[1:-1]
        logger = pl.loggers.TensorBoardLogger(
            save_dir=logger_base,
            name=os.path.join(*logdir[:-1]),
            version=logdir[-1],
        )
      
    ###########################
    # CALLBACKS
    ###########################
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
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
        save_last=not args.use_best_model,
    )
    callbacks.append(checkpoint_cb_every)

    if args.loss_type != 'pre':
        checkpoint_cb_bestk = pl.callbacks.ModelCheckpoint(
            dirpath=save_path, 
            filename="best_auc-{epoch:02d}-{val_auc:.4f}",
            save_top_k=1, 
            monitor='val_auc',
            mode='max',
            verbose=True,
            save_last=args.use_best_model,
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

    trainer = pl.Trainer(
        accelerator='gpu' if not args.debug else 'cpu',
        deterministic=True if args.seed is not None else False,
        max_epochs=args.epochs,
        check_val_every_n_epoch=int(args.evaluate_every),
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        fast_dev_run=args.debug,   # for testing training and validation
        num_sanity_val_steps=2,
    )
    trainer.fit(ensemble_model, ckpt_path=args.resume)

    if args.test is not None:
        test_models = [('test', args.test)]
    else:
        test_models = [('last', checkpoint_cb_every.best_model_path), ('best', checkpoint_cb_bestk.best_model_path)]        
    for name, cp in test_models:
        # cp = checkpoint_cb_bestk if args.use_best_model else checkpoint_cb_every
        ## validate model
        results_val = trainer.validate(
            model=ensemble_model,
            ckpt_path=cp,
            verbose=True,
        )
        ## test model
        results_test = trainer.test(
            model=ensemble_model,
            ckpt_path=cp,
            verbose=True,
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


if __name__ == '__main__':
    from arguments import args as a
    main(a)