'''
FORCE
Copyright (c) 2020-present NAVER Corp.
MIT license
'''

import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F

from .models import *
from .datasets import *

network_name_module = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet110': resnet110,
}

dataset_num_classes = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'tiny_imagenet': 200
}

def vgg_cifar_experiment(device, network_name, dataset, frac_data_for_train=0.9):
    """
    Util function to generate necessary components to train VGG network
    on CIFAR10/100 datasets.
    """

    INIT_LR = 0.1
    BATCH_SIZE = 128
    milestones=[150, 250]
    EPOCHS = 350
    WEIGHT_DECAY_RATE = 0.0005

    if network_name == 'vgg19':
        depth = 19
    elif network_name == 'vgg16':
        depth = 16
    else:
        raise NotImplementedError
    net = VGG(dataset=dataset, depth=depth).to(device)

    optimiser = optim.SGD(net.parameters(),
                              lr=INIT_LR,
                              momentum=0.9,
                              weight_decay=WEIGHT_DECAY_RATE)
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=milestones, gamma=0.1)

    train_loader, val_loader = get_cifar_train_valid_loader(
        batch_size=BATCH_SIZE,
        augment=True,
        random_seed=1,
        valid_size=1-frac_data_for_train,
        pin_memory=False,
        dataset_name=dataset
    )

    test_loader = get_cifar_test_loader(
        batch_size=BATCH_SIZE,
        pin_memory=False,
        dataset_name=dataset
    )

    loss = F.cross_entropy
    return net, optimiser, lr_scheduler, train_loader, val_loader, test_loader, loss, EPOCHS


def vgg_tiny_imagenet_experiment(device, network_name, dataset):
    """
    Util function to generate necessary components to train VGG network
    on Tiny Imagenet dataset.
    """

    INIT_LR = 0.1
    BATCH_SIZE = 128
    milestones=[150, 225]
    EPOCHS = 300
    WEIGHT_DECAY_RATE = 0.0005

    if network_name == 'vgg19':
        depth = 19
    elif network_name == 'vgg16':
        depth = 16
    else:
        raise NotImplementedError
    net = VGG(dataset=dataset, depth=depth).to(device)

    optimiser = optim.SGD(net.parameters(),
                              lr=INIT_LR,
                              momentum=0.9,
                              weight_decay=WEIGHT_DECAY_RATE)
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=milestones, gamma=0.1)

    train_loader, test_loader = get_tiny_imagenet_train_valid_loader(BATCH_SIZE,
                                                                     augment=True,
                                                                     shuffle=True,
                                                                     num_workers=8)
    val_loader = None

    loss = F.cross_entropy
    return net, optimiser, lr_scheduler, train_loader, val_loader, test_loader, loss, EPOCHS


def resnet_tiny_imagenet_experiment(device, network_name, dataset, in_planes):
    """
    Util function to generate necessary components to train resnet network
    on Tiny Imagenet dataset.
    """

    INIT_LR = 0.1
    BATCH_SIZE = 128
    milestones=[150, 225]
    EPOCHS = 300
    WEIGHT_DECAY_RATE = 0.0005

    print(network_name)
    num_classes = dataset_num_classes[dataset]
    network_name = network_name.split('stable')[-1]
    net = network_name_module[network_name](num_classes=num_classes, stable_resnet=False,
                                            in_planes=in_planes).to(device)

    optimiser = optim.SGD(net.parameters(),
                          lr=INIT_LR,
                          momentum=0.9,
                          weight_decay=WEIGHT_DECAY_RATE)
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=milestones, gamma=0.1)

    train_loader, test_loader = get_tiny_imagenet_train_valid_loader(BATCH_SIZE,
                                                                     augment=True,
                                                                     shuffle=True,
                                                                     num_workers=8)
    val_loader = None

    loss = F.cross_entropy
    return net, optimiser, lr_scheduler, train_loader, val_loader, test_loader, loss, EPOCHS


def resnet_cifar_experiment(device, network_name, dataset_name, optimiser_name="sgd",
                            frac_data_for_train=0.9, stable_resnet=False, in_planes=64):
    """
    Util function to generate necessary components to train resnet network
    on CIFAR10/100 datasets.
    """

    INIT_LR = 0.1
    BATCH_SIZE = 128
    milestones = [150, 250]
    EPOCHS = 350
    WEIGHT_DECAY_RATE = 0.0005

    print(network_name)
    num_classes = dataset_num_classes[dataset_name]
    network_name = network_name.split('stable')[-1]
    net = network_name_module[network_name](num_classes=num_classes, stable_resnet=stable_resnet,
                                            in_planes=in_planes).to(device)
    torch.backends.cudnn.benchmark = True

    if optimiser_name == "sgd":
        optimiser = optim.SGD(net.parameters(),
                              lr=INIT_LR,
                              momentum=0.9,
                              weight_decay=WEIGHT_DECAY_RATE)
    elif optimiser_name == "adam":
        optimiser = optim.Adam(net.parameters(),
                               lr=INIT_LR,
                               weight_decay=WEIGHT_DECAY_RATE)

    scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=milestones, gamma=0.1)

    train_loader, val_loader = get_cifar_train_valid_loader(
        batch_size=BATCH_SIZE,
        augment=True,
        random_seed=1,
        valid_size=1-frac_data_for_train,
        pin_memory=False,
        dataset_name=dataset_name
    )

    test_loader = get_cifar_test_loader(
        batch_size=BATCH_SIZE,
        pin_memory=False,
        dataset_name=dataset_name
    )

    loss = F.cross_entropy
    return net, optimiser, scheduler, train_loader, val_loader, test_loader, loss, EPOCHS


def train_cross_entropy(epoch, model, train_loader, optimizer, device, writer, LOG_INTERVAL=20):
    '''
    Util method for training with cross entropy loss.
    '''
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar("training/loss", loss.item(),
                              epoch*len(train_loader)+batch_idx)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))
    return train_loss / len(train_loader)
