import torch.nn as nn
import numpy as np
# from IPython import embed

def apply_prune_mask(net, keep_masks, apply_hooks=True):
    """
    Function that takes a network and a list of masks and applies it to the relevant layers.
    mask[i] == 0 --> Prune parameter
    mask[i] == 1 --> Keep parameter
    
    If apply_hooks == False, then set weight to 0 but do not block the gradient.
    This is used for FORCE algorithm that sparsifies the net instead of pruning.
    """
    
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all non-prunable modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    # List of hooks to be applied on the gradients. It's useful to save them in order to remove
    # them later
    hook_handlers = []
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)
        
        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """
            def hook(grads):
                return grads * keep_mask

            return hook

        # Step 1: Set the masked weights to zero (Biases are ignored)
        layer.weight.data[keep_mask == 0.] = 0.
        
        # Step 2: Make sure their gradients remain zero (not with FORCE)
        if apply_hooks:
            hook_handlers.append(layer.weight.register_hook(hook_factory(keep_mask)))
        
    return hook_handlers
