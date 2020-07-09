'''
FORCE
Copyright (c) 2020-present NAVER Corp.
MIT license
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from pruning.pruning_algos import iterative_pruning
from experiments.experiments import *
from pruning.mask_networks import apply_prune_mask

import os
import argparse
import random
# from IPython import embed

def parseArgs():

    parser = argparse.ArgumentParser(
                description="Training CIFAR / Tiny-Imagenet.",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--pruning_factor", type=float, default=0.01, dest="pruning_factor",
                        help='Fraction of connections after pruning')
    
    parser.add_argument("--prune_method", type=int, default=1, dest="prune_method",
                        help="""Which pruning method to use:
                                1->FORCE (default)
                                2->GRASP-It""")
    
    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        dest="dataset_name", help='Dataset to train on')
    
    parser.add_argument("--network_name", type=str, default='resnet50', dest="network_name",
                        help='Model to train')
        
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
    
    parser.add_argument("--in_planes", type=int, default=64,
                        help='''Number of input planes in Resnet. Afterwards they duplicate after
                        each conv with stride 2 as usual.''')

    return parser.parse_args()


LOG_INTERVAL = 20
REPEAT_WITH_DIFFERENT_SEED = 3 # Number of initialize-prune-train trials (minimum of 1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# New additions
args = parseArgs()


def train(seed):
    
    # Set manual seed
    torch.manual_seed(seed)
    
    if 'resnet' in args.network_name:
        stable_resnet = False
        if 'stable' in args.network_name:
            stable_resnet = True
        if 'CIFAR' in args.dataset_name:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = resnet_cifar_experiment(device, args.network_name,
                                                                   args.dataset_name, args.optimiser,
                                                                   args.frac_data_for_train,
                                                                   stable_resnet, args.in_planes)
        elif 'tiny_imagenet' in args.dataset_name: 
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = resnet_tiny_imagenet_experiment(device, args.network_name,
                                                                          args.dataset_name, args.in_planes)
            
        
    elif 'vgg' in args.network_name or 'VGG' in args.network_name:
        if 'tiny_imagenet' in args.dataset_name: 
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = vgg_tiny_imagenet_experiment(device, args.network_name,
                                                                       args.dataset_name)
        else:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = vgg_cifar_experiment(device, args.network_name,
                                                               args.dataset_name, args.frac_data_for_train)
    
    # Initialize network
    for layer in net.modules():
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
    
    ############################################################################
    ####################        Pruning at init         ########################
    ############################################################################
    pruning_factor = args.pruning_factor
    keep_masks=[]
    if pruning_factor != 1:
        print(f'Pruning network iteratively for {args.num_steps} steps')
        keep_masks = iterative_pruning(net, train_loader, device, pruning_factor,
                                       prune_method=args.prune_method,
                                       num_steps=args.num_steps,
                                       mode=args.mode, num_batches=args.num_batches)
        apply_prune_mask(net, keep_masks)
        filename = f'iter_prun_{args.num_steps}'
            
    
    ############################################################################
    ####################          Training              ########################
    ############################################################################
    evaluator = create_supervised_evaluator(net, {
        'accuracy': Accuracy(),
        'cross_entropy': Loss(loss)
    }, device)

    run_name = (args.network_name + '_' + args.dataset_name + '_spars' +
                str(1 - pruning_factor) + '_variant' + str(args.prune_method) +
                '_train-frac' + str(args.frac_data_for_train) +
                f'_steps{args.num_steps}_{args.mode}' + f'_{args.init}' +
                f'_batch{args.num_batches}' + f'_rseed_{seed}')
        
    writer_name= 'runs/' + run_name
    writer = SummaryWriter(writer_name)

    iterations = 0
    for epoch in range(0, EPOCHS):
        lr_scheduler.step()
        train_loss = train_cross_entropy(epoch, net, train_loader, optimiser, device,
                                             writer, LOG_INTERVAL=20)
        iterations +=len(train_loader)
        # Evaluate
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        # Save history
        avg_accuracy = metrics['accuracy']
        avg_cross_entropy = metrics['cross_entropy']
        writer.add_scalar("test/loss", avg_cross_entropy, iterations)
        writer.add_scalar("test/accuracy", avg_accuracy, iterations)
            
        # Save model checkpoints
        if (epoch + 1) % args.save_interval == 0:
            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            save_name = args.save_loc + run_name + '_cross_entropy_' + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)
        elif (epoch + 1) == EPOCHS:
            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            save_name = args.save_loc + run_name + '_cross_entropy_' + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), save_name)


if __name__ == '__main__':
    
    # Randomly pick a random seed for the experiment
    # Multiply the number of seeds to be sampled by 300 so there is wide range of seeds    
    seeds = list(range(300 * REPEAT_WITH_DIFFERENT_SEED))
    random.shuffle(seeds)
    
    for seed in seeds[:REPEAT_WITH_DIFFERENT_SEED]:
        train(seed)
