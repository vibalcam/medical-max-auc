import argparse


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
parser.add_argument('--loss_type', default='auc', type=str, help='loss type', choices=['auc', 'bce', 'comp'])
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial (base) learning rate')
parser.add_argument('--optimizer', default='adamw', type=str, choices=['sgd', 'adamw'])
parser.add_argument('--warmup_epochs', default=0, type=float, help='number of warmup epochs')
parser.add_argument('--lr_steps', default=[50, 75], type=int, nargs="+", help='epochs to decay learning rate by 10')

# combined loss
parser.add_argument('--lr0', default=0.02, type=float)
parser.add_argument('--betas', default=[0.9,0.9], type=float, nargs='+',)

# auc options
parser.add_argument('--epoch_decay', default=3e-3, type=float)
parser.add_argument('--margin', default=1.0, type=float)

# dataset 
parser.add_argument('--save_dir', default='./saved_models/', type=str)
parser.add_argument('--results_file', default='results', type=str)
parser.add_argument('--dataset', type=str, default="breastmnist", choices=["breastmnist", "pneumoniamnist", "chestmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d",])
parser.add_argument('--augmentations', type=str, default="basic")
parser.add_argument('--aug_args', type=str, default='gn.ra')
parser.add_argument('--sampler', type=float, default=None)

# saving
parser.add_argument('--save_every_epochs', default=5, type=int, help='number of epochs to save checkpoint')
parser.add_argument('-e', '--evaluate_every', default=5, type=float, help='evaluate model on validation set every # epochs')
parser.add_argument('--early_stopping_patience', default=None, type=int, help='patience for early stopping')
parser.add_argument('--use_best_model', action='store_true', help='use best model for evaluation')
parser.add_argument('--test', type=str, default=None, help='path to model to test')

# other model
parser.add_argument('--dropout', type=float, default=None, help='dropout rate')
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--freeze", action='store_true')
parser.add_argument("--pretrain_type", type=str, default='bce')
parser.add_argument("--type_3d", type=str, default='3d')

args = parser.parse_args()
