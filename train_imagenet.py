import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tensorboardX import SummaryWriter
from pruning.pruning_algos import iterative_pruning
from pruning.mask_networks import apply_prune_mask
from torchvision.models.vgg import vgg16_bn, vgg19_bn

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

#####################################################################
################   Arguments from pytorch code  #####################
#####################################################################
parser.add_argument('data', metavar='DIR',
                    help='path to imagenet dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

#####################################################################
################     Arguments for pruning      #####################
#####################################################################
parser.add_argument("--network_name", type=str, default='resnet50', dest="network_name",
                    help='Model to train')

parser.add_argument("--pruning_factor", type=float, default=0.1, dest="pruning_factor",
                    help='Percentage of connections retained')

parser.add_argument("--prune_method", type=int, default=1, dest="prune_method",
                        help="""Which pruning method to use:
                                1->FORCE (default)
                                2->GRASP-It""")

parser.add_argument("--num_steps", type=int, default=10,
                    help='Number of steps to use with iterative pruning')
    
parser.add_argument("--mode", type=str, default='exp',
                    help='Mode of creating the iterative pruning steps one of "linear" or "exp".')
    
parser.add_argument("--num_batches", type=int, default=1,
                    help='''Number of batches to be used when computing the gradient.
                    If set to -1 they will be averaged over the whole dataset.''')
    
parser.add_argument("--save_interval", type=int, default=50,
                    dest="save_interval", help="Number of epochs between model checkpoints.")
    
parser.add_argument("--save_loc", type=str, default='saved_models/',
                    dest="save_loc", help='Path where to save the model')
    
parser.add_argument("--opt", type=str, default='sgd',
                    dest="optimiser",
                    help='Choice of optimisation algorithm')
    
parser.add_argument("--saved_model_name", type=str, default="cnn.model",
                    dest="saved_model_name", help="Filename of the pre-trained model")
    
parser.add_argument("--frac-train-data", type=float, default=0.9, dest="frac_data_for_train",
                    help='Fraction of data used for training (only applied in CIFAR)')
    
parser.add_argument("--init", type=str, default='normal_kaiming',
                    help='Which initialization method to use')

seed = 1 # We fix the random seed
device = torch.device("cuda")

#############################################################################

def main():
    args = parser.parse_args()
    args.arch = args.network_name
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    if args.network_name == 'vgg19':
        model = vgg19_bn(pretrained=False)
    elif args.network_name == 'vgg16':
        model = vgg16_bn(pretrained=False)
    elif 'resnet' in args.network_name:
        model = models.__dict__[args.arch](pretrained=False)
    else:
        raise NotImplementedError
    
    # Initialize network
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if args.init == 'normal_kaiming':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif args.init == 'normal_kaiming_fout':    
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu', mode='fan_out')
            elif args.init == 'normal_xavier':    
                nn.init.xavier_normal_(layer.weight)
            elif args.init == 'orthogonal':    
                nn.init.orthogonal_(layer.weight)
            else:
                raise ValueError(f"Unrecognised initialisation parameter {args.init}")

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    #############################
    ####    Pruning code     ####
    #############################
    
    pruning_factor = args.pruning_factor
    keep_masks=[]
    filename = ''
    if pruning_factor != 1:
        print(f'Pruning network iteratively for {args.num_steps} steps')
        keep_masks = iterative_pruning(model, train_loader, device, pruning_factor,
                                       prune_method=args.prune_method,
                                       num_steps=args.num_steps,
                                       mode=args.mode, num_batches=args.num_batches)
                
        apply_prune_mask(model, keep_masks)
      
    # File where to save training history
    run_name = (args.network_name + '_IMAGENET' + '_spars' +
                str(1 - pruning_factor) + '_variant' + str(args.prune_method) +
                '_train-frac' + str(args.frac_data_for_train) +
                f'_steps{args.num_steps}_{args.mode}' + f'_{args.init}' +
                f'_batch{args.num_batches}' + f'_rseed_{seed}')
    writer_name= 'runs/' + run_name
    writer = SummaryWriter(writer_name)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    iterations = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        
        # Evaluate on validation set
        iterations = epoch * len(train_loader)
        acc1 = validate(val_loader, model, criterion, args, writer, iterations)

        # Save checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if (epoch + 1) % 5 == 0:
                if not os.path.exists('saved_models/'):
                    os.makedirs('saved_models/')
                save_name = 'saved_models/' + run_name + '_cross_entropy_' + str(epoch + 1) + '.model'
                torch.save(model.state_dict(), save_name)
            elif (epoch + 1) == args.epochs:
                if not os.path.exists('saved_models/'):
                    os.makedirs('saved_models/')
                save_name = 'saved_models/' + run_name + '_cross_entropy_' + str(epoch + 1) + '.model'
                torch.save(model.state_dict(), save_name)
        

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    total_batches = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            writer.add_scalar("training/loss", losses.avg, epoch * total_batches + i)


def validate(val_loader, model, criterion, args, writer, iterations):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        writer.add_scalar("test/loss", losses.avg, iterations)
        writer.add_scalar("test/accuracy", top1.avg, iterations)
        writer.add_scalar("test/top5", top5.avg, iterations)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 at 1/3 and 2/3 of training"""
    interval = args.epochs // 3
    lr = args.lr * (0.1 ** (epoch // interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
