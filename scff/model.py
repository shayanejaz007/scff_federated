"""
SCFF Model Building Utilities
Exact implementation matching original SCFF_CIFAR.py
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import json
from typing import List, Tuple, Dict, Any, Optional

from .layers import Conv2d


def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p=1):
    """
    Generates positive and negative inputs for SCFF.

    Args:
        batch_pos1 (torch.Tensor): First set of samples of shape (batch_size, ...).
        batch_pos2 (torch.Tensor): Second set of samples, typically an augmented version 
                                   of batch_pos1 with the same shape or the same with batch_pos1.
        p (int, optional): Number of negative samples per positive sample. Default is 1.

    Returns:
        tuple: 
            - batch_pos (torch.Tensor): Concatenated positive samples of shape (batch_size, 2 * feature_dim).
            - batch_negs (torch.Tensor): Concatenated negative samples of shape (batch_size * p, 2 * feature_dim).
    """
    batch_size = len(batch_pos1)

    batch_pos = torch.cat((batch_pos1, batch_pos2), dim=1)

    # Create negative samples by permutation
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p, batch_size - 1)]
    labels = torch.arange(batch_size)

    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labels + i) % batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim=1)
        batch_negs.append(batch_neg)
    
    return batch_pos, torch.cat(batch_negs)


def create_layer(layer_config: Dict, opt_config: Dict, load_params: bool, device: str):
    """
    Create a single SCFF layer with its pooling and optimizer.
    
    Args:
        layer_config: Layer configuration from config.json
        opt_config: Optimizer configuration from config.json
        load_params: Whether to load pretrained parameters
        device: Device to place the layer on
        
    Returns:
        Tuple of (net, pool, extra_pool, optimizer, scheduler)
    """
    layer_num = layer_config['num'] - 1

    net = Conv2d(
        layer_config["ch_in"], 
        layer_config["channels"], 
        (layer_config["kernel_size"], layer_config["kernel_size"]),
        pad=layer_config["pad"], 
        norm="stdnorm", 
        padding_mode=layer_config["padding_mode"], 
        act=layer_config["act"]
    )
    
    if load_params:
        net.load_state_dict(torch.load(f'./results/params_CIFAR_l{layer_num}.pth', map_location='cpu'))
        for param in net.parameters():
            param.requires_grad = False

    if layer_config["pooltype"] == 'Avg':
        pool = nn.AvgPool2d(
            kernel_size=layer_config["pool_size"], 
            stride=layer_config["stride_size"], 
            padding=layer_config["padding"], 
            ceil_mode=True
        )
    else:
        pool = nn.MaxPool2d(
            kernel_size=layer_config["pool_size"], 
            stride=layer_config["stride_size"], 
            padding=layer_config["padding"], 
            ceil_mode=True
        )
    
    extra_pool = nn.AvgPool2d(
        kernel_size=layer_config["extra_pool_size"], 
        stride=layer_config["extra_pool_size"], 
        padding=0, 
        ceil_mode=True
    )
    
    net.to(device)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])

    return net, pool, extra_pool, optimizer, scheduler


def build_layers_from_config(
    config_path: str, 
    dataset: str, 
    NL: int, 
    device: str, 
    load_params: bool = False,
    freeze_layer: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build all layers from config file.
    
    Args:
        config_path: Path to config.json
        dataset: Dataset name (e.g., 'CIFAR')
        NL: Number of layers to build
        device: Device to place layers on
        load_params: Whether to load pretrained parameters
        freeze_layer: Layer index up to which to freeze (exclusive)
        
    Returns:
        Dictionary containing nets, pools, extra_pools, optimizers, schedulers, and hyperparameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_config = config[dataset]
    
    nets = []
    pools = []
    extra_pools = []
    optimizers = []
    schedulers = []
    threshold1 = []
    threshold2 = []
    lamda = []
    period = []
    
    if freeze_layer is None:
        freeze_layer = NL - 1
    
    for i, (layer_config, opt_config) in enumerate(
        zip(dataset_config['layer_configs'][:NL], dataset_config['opt_configs'][:NL])
    ):
        # Determine if we should load params for this layer
        should_load = load_params and (i < freeze_layer)
        
        net, pool, extra_pool, optimizer, scheduler = create_layer(
            layer_config, opt_config, load_params=should_load, device=device
        )
        
        nets.append(net)
        pools.append(pool)
        extra_pools.append(extra_pool)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        threshold1.append(opt_config['th1'])
        threshold2.append(opt_config['th2'])
        lamda.append(opt_config['lamda'])
        period.append(opt_config['period'])
    
    # CRITICAL: Set concat flags per layer from each layer's own config
    # The concat list at index i in layer i's config indicates which layers use concat
    # For layer i, we use layer_config['concat'][i] to get this layer's concat flag
    for i, (net, layer_config) in enumerate(zip(nets, dataset_config['layer_configs'][:NL])):
        # layer_config['concat'] is a list where index i indicates concat for layer i
        net.concat = bool(layer_config['concat'][i])
    
    return {
        'nets': nets,
        'pools': pools,
        'extra_pools': extra_pools,
        'optimizers': optimizers,
        'schedulers': schedulers,
        'threshold1': threshold1,
        'threshold2': threshold2,
        'lamda': lamda,
        'period': period,
        'config': dataset_config,
        'out_dropout': dataset_config['opt_configs'][NL-1].get('out_dropout', 0.2)
    }


def get_layer_concat_flags(config_path: str, dataset: str, NL: int) -> List[bool]:
    """
    Get concat flags for each layer from config.
    
    Args:
        config_path: Path to config.json
        dataset: Dataset name
        NL: Number of layers
        
    Returns:
        List of concat flags for each layer
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_config = config[dataset]
    concat_flags = []
    
    for i, layer_config in enumerate(dataset_config['layer_configs'][:NL]):
        # Each layer's concat list contains flags for all layers up to and including itself
        concat_flags.append(bool(layer_config['concat'][i]))
    
    return concat_flags
