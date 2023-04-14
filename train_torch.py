import argparse
import gc
import math
import os
import warnings
from matplotlib import pyplot as plt

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
from libauc.metrics import pauc_roc_score
from models.dataset import get_validation_data

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG


parser = argparse.ArgumentParser(description='')
parser.add_argument('-a', '--arch', type=str, default='attention')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning-rate', default=5e-6, type=float,
                    metavar='LR', help='initial (base) learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--debug', action='store_true', help='To debug code')

# additional configs:
parser.add_argument('--pretrained', default=None, type=str)

parser.add_argument('--loss_type', default='bce', type=str,
                    help='loss type of pretrained (default: bce)')
parser.add_argument('--gammas', default=[0.9,0.9], type=float, nargs='+')
parser.add_argument('--lbda', default=0.5, type=float)
parser.add_argument('--tau', default=1.0, type=float)
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['sgd', 'adamw'],
                    help='optimizer used (default: sgd)')
parser.add_argument('--warmup-epochs', default=1, type=float, help='number of warmup epochs')

# dataset 
parser.add_argument('--save_dir', default='./saved_models/', type=str) 

# saving
parser.add_argument('--save_every_epochs', default=1, type=int,
                    help='number of epochs to save checkpoint')
parser.add_argument('-e', '--evaluate_every', default=0.25, type=float,
                    help='evaluate model on validation set every # epochs')
parser.add_argument('--early_stopping_patience', default=100, type=int,
                    help='patience for early stopping')
parser.add_argument('-s', '--save', type=str, default='defender/att.pkl', help='file where to save model')

# # model
# # parser.add_argument("--hidden_dim", type=int, default=768)
# parser.add_argument("--hidden_dim", type=int, nargs='+', default=[512], help="Hidden dimension. Only the first number will be used for attention")
# parser.add_argument("--n_layers", type=int, default=12, help="Number of attention layers")
# parser.add_argument("--num_heads", type=int, default=8)
# parser.add_argument("--dropout_rate", type=float, default=0.7)
# parser.add_argument("--att_dropout", type=float, default=0.5)
# # parser.add_argument("--n_tries", type=int, default=100, help="Number of thresholds to test")

# parser.add_argument('-sk', "--skip_training", type=str, default=None)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30"


"""
todo!

set up datasets
set up model
set up auc and testing
"""


class DetectorModel(pl.LightningModule):
    def __init__(
        self,
        use_features,
        features_dim,
        args,
        num_batches,
        pos_samples=None,
        textual_idx=None,
        val_datasets=1,
        **kwargs,
    ):
        super().__init__()
        # random input to build computational graph
        # self.example_input_array = torch.zeros((1, ))

        # save hyperparameters as attribute
        self.save_hyperparameters(ignore=['model'])
        self.pos_samples = pos_samples
        # self.threshold = 0.5
        self.num_batches = num_batches
        self.first = False
        
        self.args = args
        if args.arch == 'attention':
            pass
        else:
            raise NotImplementedError()
        print(f"Using model: {self.model.__class__.__name__}")

        self.batch_size = self.args.batch_size
        ## infer learning rate
        self.init_lr = self.args.learning_rate
        # self.init_lr = self.args.lr * self.batch_size / 256
        self.lr = self.init_lr
        print('initial learning rate:', self.lr)

        self.scheduler_metric = None

        ##################
        # METRICS
        ##################
        if args.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'auc':
            self.criterion = AUCMLoss()
        elif args.loss_type == 'pauc':
            # todo check values
            self.criterion = tpAUC_KL_Loss(self.pos_samples, Lambda=args.lbda, tau=args.tau)
        else:
            raise NotImplementedError()

        self.train_auc = torchmetrics.AUROC(task='binary')
        self.train_roc = torchmetrics.ROC(task='binary')
        self.val_auc = torchmetrics.AUROC(task='binary')
        self.val_roc = torchmetrics.ROC(task='binary', thresholds=None)

    def configure_optimizers(self):
        ## optimizer
        if self.args.loss_type == 'auc':
            optimizer = PESG(
                self.model, 
                loss_fn=self.criterion, 
                lr=self.lr, 
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.loss_type == 'pauc':
            optimizer = SOTAs(
                self.parameters(), 
                loss_fn=self.criterion, 
                lr=self.lr, 
                mode='adam',
                gammas=self.args.gammas,
                weight_decay=self.args.weight_decay,
            )
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), self.lr,
                                                weight_decay=self.args.weight_decay,
                                                momentum=self.args.momentum)
            elif self.args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(self.parameters(), self.lr,
                                        weight_decay=self.args.weight_decay)
            else:
                raise NotImplementedError("Optimizer not implemented")

        ## for lr scheduler
        self.scheduler_metric = SchedulerMetricWithWarmup(optimizer, self.args.warmup_epochs, self.lr, patience=8, maximize=True)

        ## lr scheduler
        # scheduler = get_linear_schedule_with_warmup(optimizer,num_training_steps=n_epochs, num_warmup_steps=100)
        # plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4, verbose=True)
        # def lambda_lr(epoch):
        #     (epoch+1) / warmup_steps if epoch < warmup_steps else warmup_factor
        # warm_up_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lambda_lr,
        #     verbose=True,
        # )
        # lr_scheduler_config = {
        #     "scheduler": plateau_scheduler,
        #     "interval": "step",
        #     "frequency": self.num_batches * self.args.evaluate_every,
        #     "monitor": "val_auc",
        #     "strict": True,
        #     "name": "scheduler",
        # }

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': lr_scheduler_config,
        }
    
    # def lr_scheduler_step(self, scheduler, metric):
    #     pass
        
    def forward(self, x):
        return self.model(x, given_idx=True)

    def on_train_epoch_start(self):
        self.train_preds = []
        self.train_true = []

    def training_step(self, batch, batch_idx):
        # adjust_learning_rate(self.optimizers(), self.init_lr, self.current_epoch + batch_idx/self.num_batches, self.args)
        self.scheduler_metric.step(self.current_epoch + batch_idx/self.num_batches)

        # run through model
        x, target, index = batch

        output = self(x)
        if output.isnan().any():
            warnings.warn("Nan values being generated")
        
        if self.args.loss_type=='pauc':
            # loss = self.criterion(torch.sigmoid(output).float(), target, index.long())
            loss = self.criterion(torch.sigmoid(output).float(), target, index[:self.pos_samples].long())
        elif self.args.loss_type=='mse':
            loss = self.criterion(torch.sigmoid(output).float(), target.float())
        else:
            loss = self.criterion(output, target)
        if loss.isnan():
            warnings.warn("Getting nan loss")

        # calculate metrics
        y_pred = torch.sigmoid(output).detach().float()
        self.train_auc(y_pred, target.int().detach())
        # self.train_roc.update(y_pred, target.int().detach())

        pred = y_pred.cpu().numpy()
        true = target.int().detach().cpu().numpy()
        # self.train_preds.append(pred)
        # self.train_true.append(true)
        train_pauc = pauc_roc_score(true, pred, self.max_fpr, self.min_tpr)

        # log loss for training step and average loss for epoch
        self.log_dict({
            "train_loss": loss,
            "train_auc": self.train_auc,
            # "train_score": self.train_score,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_pauc', train_pauc, True, True, True, False)

        ## compute gradient and do SGD step
        ## automatically done by lightning
        ## can be disabled (https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html)
        # optimizer.zero_grad()
        # outputs = model(input)
        # loss = loss_f(output, labels)
        # loss.backward()
        # optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()
        
        return loss
    
    def on_train_epoch_end(self, *args, **kwargs):
        # self.log_dict({
        #     "train_pauc": train_pauc,
        #     # "train_score": score[idx],
        #     # "train_fpr": fpr[idx],
        #     # "train_tpr": tpr[idx],
        # }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_true = []
        self.train_preds = []
        self.train_roc.reset()
        torch.cuda.empty_cache()
        gc.collect()
    
    def on_validation_start(self) -> None:
        self.val_preds = []
        self.val_true = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):   
        # run through model
        x, target = batch[:2]
        output = self(x)

        # calculate metrics
        y_pred = torch.sigmoid(output).float().detach()
        target = target.int().detach()
        if dataloader_idx != 0:
            self.val_auc.update(y_pred, target)
            self.val_roc.update(y_pred, target)
            self.val_preds.append(y_pred.cpu().numpy())
            self.val_true.append(target.cpu().numpy())

            self.val_metrics[dataloader_idx-1][0].update(y_pred, target)
            self.val_metrics[dataloader_idx-1][1].update(y_pred, target)
        else:
            self.val_dataset_auc(y_pred, target)
            self.log_dict({
                "val_dataset_auc": self.val_dataset_auc,
            }, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_validation_epoch_end(self):
        if len(self.val_roc.target) == 0:
            return

        fpr, tpr, thr = [k.detach().cpu() for k in self.val_roc.compute()]
        self.logger.experiment.add_figure('val_roc', plot_roc(tpr, fpr), global_step=self.global_step)

        preds = np.concatenate(self.val_preds, axis=0)
        preds_nan = np.isnan(preds)
        preds[preds_nan] = 1
        try:
            val_pauc = pauc_roc_score(np.concatenate(self.val_true, axis=0), preds, self.max_fpr, self.min_tpr)
        except:
            val_pauc = 0

        self.threshold = 1
        for idx, met in enumerate(self.val_metrics):
            i_fpr, i_tpr, i_thr = [k.detach().cpu() for k in met[1].compute()]
            if (i_fpr == 0).all() or (torch.cat(met[1].target, dim=0) == 1).all():
                continue
            i_idx = (i_fpr <= self.max_fpr).nonzero().max()
            c_thr = i_thr[i_idx]
            if not c_thr.isnan() and c_thr < self.threshold:
                self.threshold = c_thr
                print(f"New threshold: {c_thr}, with {i_fpr[i_idx]} on dataset {idx}")

        if self.scheduler_metric is not None:
            self.scheduler_metric.update(self.val_auc.compute())

        self.log_dict({
            "val_auc": self.val_auc,
            "val_pauc": val_pauc,
            "val_fpr": fpr[(thr - c_thr).abs().argmin()],
            "val_tpr": tpr[(thr - c_thr).abs().argmin()],
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # reset metrics
        for met in self.val_metrics:
            met[0].reset()
            met[1].reset()
        self.val_roc.reset()
        self.val_preds = []
        self.val_true = []
        torch.cuda.empty_cache()
        gc.collect()

    # def on_test_start(self) -> None:
    #     self.preds = []
    #     self.targets = []

    # def test_step(self, batch, batch_idx):
    #     # run through model
    #     images, target = batch
    #     output = self(images)
    #     y_pred = torch.sigmoid(output)

    #     self.preds.append(y_pred.detach().cpu())
    #     self.targets.append(target.detach().cpu())

    # def on_test_epoch_end(self) -> None:
    #     self.preds = torch.cat(self.preds, dim=0)
    #     self.targets = torch.cat(self.targets, dim=0)
    #     torch.save(self.preds, os.path.join(self.save_path, "preds.pt"))
    #     torch.save(self.targets, os.path.join(self.save_path, "targets.pt"))


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


class SchedulerMetricWithWarmup:
    def __init__(self, optimizer, warmup_epochs, init_lr, factor=0.1, min_lr=0, patience=10, maximize=True):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
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

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

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
            print(f"Metric did not improve: {self.curr_patience} epochs left to reduce lr")

        if self.curr_patience == 0:
            self.curr_patience = self.patience
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.factor
                if param_group['lr'] < self.min_lr:
                    param_group['lr'] = self.min_lr
                    print(f"Minimum lr {self.min_lr} reached")
                else:
                    print(f"New lr: {param_group['lr']}")


def plot_roc(tpr,fpr):
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    return fig


def main():
    ###########################
    # PARAMETERS
    ###########################
    args = parser.parse_args()

    if args.debug:
        args.workers = 0
    
    # seed for reproducibility
    if args.seed is not None:
        warnings.warn(f"You have seeded training with seed {args.seed}")
        pl.seed_everything(args.seed, workers=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        warnings.warn(f"You have not seeded training")

    if not torch.cuda.is_available():
        warnings.warn("No GPU available: training will be extremely slow")

    ###########################
    # DATASET
    ###########################

    # dataset1 = SparseDataset('data/train.pkl', use_features=use_features, use_ember=False)
    # dataset2 = SparseDataset('data/test.pkl', use_features=use_features, use_ember=False)
    # full_dataset = data.ConcatDataset([dataset1, dataset2])
    # train_dataset, val_dataset = split_generator_dataset(full_dataset, [0.8, 0.2], args.seed)


    # todo! get datasets


    train_dataloader = DataLoader(
        train_dataset, 
        sampler=sampler,
        batch_size=args.batch_size, 
        # shuffle=True,
        shuffle=False,
        num_workers=args.workers, 
        # pin_memory=not args.debug,
        drop_last=not args.debug,
    )

    val_datasets = get_validation_data(gw_samples=6,mw_samples=8,data_path='data/testData',extractor= feature_extractor, use_features=use_features, textual_idx=use_textual, cache=use_cache)

    val_dataloader = [DataLoader(
        k, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=False,
        drop_last=False,
    ) for k in val_datasets]


    ###########################
    # MODEL
    ###########################

    base_dir = os.path.join(args.arch, args.loss_type, str(use_features))
    if args.resume is None:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=args.save_dir,
            name=base_dir, 
            # log_graph=True,
        )
    else:
        logdir = args.resume.split('/')[1:-1]
        logger = pl.loggers.TensorBoardLogger(
            save_dir=args.save_dir,
            name=os.path.join(*logdir[:-1]),
            version=logdir[-1],
            # log_graph=True,
        )

    # adjust path for pretrain according to model
    # args.pretrained = os.path.join(args.save_dir, base_dir, "pretrain", args.pretrained)

    # load pretrained model
    # pretrained_model = SogModel.load_from_checkpoint(args.pretrained).model

    print("-"*10)
    print(train_dataset.feature_dims())
    print("-"*10)
    # task to do
    if args.pretrained is not None:
        model_task = DetectorModel.load_from_checkpoint(
            args.pretrained,
            args=args,
            pos_samples=sampler.pos_len,
            num_batches=len(train_dataloader),
            val_datasets=len(val_dataloader)-1 if isinstance(val_dataloader, list) else 1,
        )
    else:
        model_task = DetectorModel(
            use_features=use_features,
            features_dim=train_dataset.feature_dims(),
            textual_idx=use_textual,
            args=args,
            # pos_samples=num_pos,
            pos_samples=sampler.pos_len,
            num_batches=len(train_dataloader),
            val_datasets=len(val_dataloader)-1 if isinstance(val_dataloader, list) else 1,
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
        filename="last",
        monitor="step",
        mode="max",
        save_top_k=1,
        every_n_epochs=args.save_every_epochs,
        # save_on_train_epoch_end=True,                         # when using a training metric
        # train_time_interval=,
        # every_n_train_steps=,
        # save_last=False,                                        # save last might be useful to have
    )
    callbacks.append(checkpoint_cb_every)

    checkpoint_cb_bestk = pl.callbacks.ModelCheckpoint(
        dirpath=save_path, 
        filename="best_auc",
        save_top_k=2, 
        monitor='val_auc',
        mode='max',
        verbose=True,
        # save_on_train_epoch_end=False,     # when using a training metric
        # save_last=False,
    )
    callbacks.append(checkpoint_cb_bestk)

    # early stopping
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
    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        accelerator='gpu' if not args.debug else 'cpu',
        deterministic="warn" if args.seed is not None else False,
        # precision="16-mixed",   # reduce memory, can improve performance but might lead to unstable training
        
        max_epochs=args.epochs,
        # max_time="00:1:00:00",
        # max_steps=,

        # check_val_every_n_epoch=args.evaluate_every,
        val_check_interval=args.evaluate_every,
        logger=logger,
        log_every_n_steps=10,
        callbacks=callbacks,

        fast_dev_run=args.debug,   # for testing training and validation
        num_sanity_val_steps=2,
        limit_train_batches=1.0 if not args.debug else 0.01,  # to test what happens after an epoch
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

    ## test model
    # trainer.test(
    #     model=model,
    #     dataloaders=test_dataloader, 
    #     ckpt_path=os.path.join(save_path, "best.ckpt"),
    #     verbose=True,
    # )


if __name__ == '__main__':
    main()
