'''
FORCE
Copyright (c) 2020-present NAVER Corp.
MIT license
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
# from IPython import embed

from .mask_networks import apply_prune_mask


####################################################
############### Get saliencies    ##################
####################################################

def get_average_gradients(net, train_dataloader, device, num_batches=-1):
    """
    Function to compute gradients and average them over several batches.
    
    num_batches: Number of batches to be used to approximate the gradients. 
                 When set to -1, uses the whole training set.
    
    Returns a list of tensors, with gradients for each prunable layer.
    """
    
    # Prepare list to store gradients
    gradients = []
    for layer in net.modules():
        # Select only prunable layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            gradients.append(0)
    
    # Take a whole epoch
    count_batch = 0
    for batch_idx in range(len(train_dataloader)):
        inputs, targets = next(iter(train_dataloader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Compute gradients (but don't apply them)
        net.zero_grad()
        outputs = net.forward(inputs)
        loss = F.nll_loss(outputs, targets)
        loss.backward()
        
        # Store gradients
        counter = 0
        for layer in net.modules():
            # Select only prunable layers
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                gradients[counter] += layer.weight.grad
                counter += 1
        count_batch += 1
        if batch_idx == num_batches - 1:
            break
    avg_gradients = [x / count_batch for x in gradients] 
        
    return avg_gradients

    
def get_average_saliencies(net, train_dataloader, device, prune_method=1, num_batches=-1):
    """
    Get saliencies with averaged gradients.
    
    num_batches: Number of batches to be used to approximate the gradients. 
                 When set to -1, uses the whole training set.
    
    prune_method: Which method to use to prune the layers, refer to https://arxiv.org/abs/2006.09081.
                   1: Use FORCE (default).
                   2: Use GRASP-It.
    
    Returns a list of tensors with saliencies for each weight.
    """
     
    def pruning_criteria(number):
        if number == 1:
            # FORCE method (which approximates to using SNIP at each iteration)
            result = torch.abs(layer_weight * layer_weight_grad)
        elif number == 2:
            # GRASP-It method (which corresponds to applying the gradient approximation iteratively)
            result = layer_weight_grad**2 # Custom gradient norm approximation
        return result

    gradients = get_average_gradients(net, train_dataloader, device, num_batches)
    saliency = []
    idx = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer_weight = layer.weight
            layer_weight_grad = gradients[idx]
            idx += 1
            saliency.append(pruning_criteria(prune_method))
                
    return saliency
    
###################################################
############# Iterative pruning ###################
###################################################
        
def get_mask(saliency, pruning_factor):
    """ 
    Given a list of saliencies and a pruning factor (sparsity),
    returns a list with binary tensors which correspond to pruning masks.
    """
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in saliency])

    num_params_to_keep = int(len(all_scores) * pruning_factor)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for m in saliency:
        keep_masks.append((m >= acceptable_score).float())
    return keep_masks

def iterative_pruning(net, train_dataloader, device, pruning_factor=0.1, prune_method=1, num_steps=10,
                      mode='exp', num_batches=1):
    """
    Function to gradually remove weights from a network, recomputing the saliency at each step.
    
    pruning_factor: Fraction of remaining weights (globally) after pruning. 
    
    prune_method: Which method to use to prune the layers, refer to https://arxiv.org/abs/2006.09081.
                   1: Use FORCE (default).
                   2: Use GRASP-It.
                   
    num_steps: Number of iterations to do when pruning progrssively (should be >= 1).  
                   
    mode: Mode of choosing the sparsity decay schedule. One of 'exp', 'linear'
                   
    num_batches: Number of batches to be used to approximate the gradients (should be -1 or >= 1). 
                 When set to -1, uses the whole training set.
                 
    Returns a list of binary tensors which correspond to the final pruning mask.
    """
    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)
    percentages = []
    
    # Choose a decay mode for sparsity
    if mode == 'linear':
        pruning_steps = [1 - ((x + 1) * (1 - pruning_factor) / num_steps) for x in range(num_steps)]
        
    elif mode == 'exp':
        pruning_steps = [np.exp(0 - ((x + 1) * (0 - np.log(pruning_factor)) / num_steps)) for x in range(num_steps)]
   
    mask = None
    hook_handlers = None
    
    for perc in pruning_steps:
        saliency = []
        saliency = get_average_saliencies(net, train_dataloader, device,
                                          prune_method=prune_method, num_batches=num_batches)
        torch.cuda.empty_cache()
        
        # Make sure all saliencies of previously deleted weights is minimum so they do not 
        # get picked again.
        if mask is not None:
            min_saliency = get_minimum_saliency(saliency)
            for ii in range(len(saliency)):
                saliency[ii][mask[ii] == 0.] = min_saliency
        
        if hook_handlers is not None:
            for h in hook_handlers:
                h.remove()
        mask = []
        mask = get_mask(saliency, perc)
        hook_handlers = apply_prune_mask(net, mask)
        
        p = check_global_pruning(mask)
        print(f'Global pruning {round(float(p),5)}')
    
    return mask 

def check_global_pruning(mask):
    "Compute fraction of unpruned weights in a mask"
    flattened_mask = torch.cat([torch.flatten(x) for x in mask])
    return flattened_mask.mean()

def get_minimum_saliency(saliency):
    "Compute minimum value of saliency globally"
    flattened_saliency = torch.cat([torch.flatten(x) for x in saliency])
    return flattened_saliency.min()

def get_maximum_saliency(saliency):
    "Compute maximum value of saliency globally"
    flattened_saliency = torch.cat([torch.flatten(x) for x in saliency])
    return flattened_saliency.max()


####################################################################
######################    UTILS    #################################
####################################################################

def get_force_saliency(net, mask, train_dataloader, device, num_batches):
    """
    Given a dense network and a pruning mask, compute the FORCE saliency.
    """
    net = copy.deepcopy(net)
    apply_prune_mask(net, mask, 0, apply_hooks=True)
    saliencies = get_average_saliencies(net, train_dataloader, device,
                                        1, num_batches=num_batches)
    torch.cuda.empty_cache()
    s = sum_unmasked_saliency(saliencies, mask)
    torch.cuda.empty_cache()
    return s

def sum_unmasked_saliency(variable, mask):
    "Util to sum all unmasked (mask==1) components"
    V = 0
    for v, m in zip(variable, mask):
        V += v[m > 0].sum()
    return V.detach().cpu()

def get_gradient_norm(net, mask, train_dataloader, device, num_batches):
    "Given a dense network, compute the gradient norm after applying the pruning mask."
    net = copy.deepcopy(net)
    apply_prune_mask(net, mask)
    gradients = get_average_gradients(net, train_dataloader, device, num_batches)
    torch.cuda.empty_cache()
    norm = 0
    for g in gradients:
        norm += (g**2).sum().detach().cpu().numpy()
    return norm
