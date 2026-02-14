#!/usr/bin/env python3
"""
Baseline SCFF Training on 10,000 samples

For fair comparison with federated training.
Uses the same 10,000 sample subset and same total compute budget.
"""
import argparse
import json
import torch
from torch.utils.data import DataLoader, Subset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scff.data import DualAugmentCIFAR10, DualAugmentCIFAR10_test
from scff.model import build_layers_from_config, get_pos_neg_batch_imgcats
from scff.layers import stdnorm
from scff.eval import EvaluationConfig, evaluate_model


def get_arguments():
    parser = argparse.ArgumentParser(description="Baseline SCFF Training (10k samples)")
    parser.add_argument("--dataset", type=str, default="CIFAR")
    parser.add_argument("--NL", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--dataset_size", type=int, default=10000, help="Number of training samples")
    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    
    print("=" * 60)
    print("Baseline SCFF Training (10k samples)")
    print("=" * 60)
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.epochs is None:
        epochs = config[args.dataset]['opt_configs'][args.NL - 1]['epochs']
    else:
        epochs = args.epochs
    print(f"Training for {epochs} epochs")
    
    # Load data (10k subset)
    print("\nLoading data...")
    train_dataset = DualAugmentCIFAR10(root='./data', train=True, download=True, augment="no")
    
    # Select 10k samples deterministically
    torch.manual_seed(args.seed)
    all_indices = torch.randperm(len(train_dataset)).tolist()[:args.dataset_size]
    train_subset = Subset(train_dataset, all_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=0)
    
    print(f"Training set: {len(train_subset)} samples")
    
    # Test data
    test_dataset = DualAugmentCIFAR10_test(root='./data', aug=False, train=False, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    # Supervised training data for evaluation
    sup_train_dataset = DualAugmentCIFAR10_test(root='./data', aug=True, train=True, download=True)
    sup_train_subset = Subset(sup_train_dataset, all_indices)
    sup_train_loader = DataLoader(sup_train_subset, batch_size=64, shuffle=True, num_workers=0)
    
    # Build model
    print("\nBuilding model...")
    torch.manual_seed(args.seed)
    model_components = build_layers_from_config(
        config_path=args.config,
        dataset=args.dataset,
        NL=args.NL,
        device=device,
        load_params=False,
        freeze_layer=args.NL - 1
    )
    
    nets = model_components['nets']
    pools = model_components['pools']
    extra_pools = model_components['extra_pools']
    optimizers = model_components['optimizers']
    schedulers = model_components['schedulers']
    threshold1 = model_components['threshold1']
    threshold2 = model_components['threshold2']
    lamda = model_components['lamda']
    period = model_components['period']
    out_dropout = model_components['out_dropout']
    
    print("Layer concat flags:")
    for i, net in enumerate(nets):
        print(f"  Layer {i}: concat = {net.concat}")
    
    # Training parameters
    dims_in = (1, 2, 3)
    p = 1
    a = 1.0
    b = 1.0
    freeze_layer = args.NL - 1
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    nbbatches = 0
    firstpass = True
    Dims = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch}")
        nets[-1].train()
        
        total_gpos = 0.0
        total_gneg = 0.0
        num_batches = 0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            
            for i in range(args.NL):
                if nets[i].concat:
                    x = stdnorm(x, dims=dims_in)
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)
                
                x = nets[i](x)
                x_neg = nets[i](x_neg)
                
                yforgrad = nets[i].relu(x).pow(2).mean([1])
                yforgrad_neg = nets[i].relu(x_neg).pow(2).mean([1])
                
                UNLAB = (i >= freeze_layer)
                
                if UNLAB:
                    optimizers[i].zero_grad()
                    
                    loss = (
                        torch.log(1 + torch.exp(
                            a * (-yforgrad + threshold1[i])
                        )).mean([1, 2]).mean()
                        + torch.log(1 + torch.exp(
                            b * (yforgrad_neg - threshold2[i])
                        )).mean([1, 2]).mean()
                        + lamda[i] * torch.norm(yforgrad, p=2, dim=(1, 2)).mean()
                    )
                    
                    loss.backward()
                    optimizers[i].step()
                    
                    nbbatches += 1
                    if nbbatches % period[i] == 0:
                        schedulers[i].step()
                        print(f'nbbatches {nbbatches} lr: {schedulers[i].get_last_lr()[0]}')
                
                x = pools[i](nets[i].act(x)).detach()
                x_neg = pools[i](nets[i].act(x_neg)).detach()
                
                if firstpass:
                    _, c, h, w = x.shape
                    Dims.append(c * h * w)
                
                gpos = torch.mean(yforgrad.mean([1, 2])).item()
                gneg = torch.mean(yforgrad_neg.mean([1, 2])).item()
            
            firstpass = False
            total_gpos += gpos
            total_gneg += gneg
            num_batches += 1
        
        avg_gpos = total_gpos / num_batches
        avg_gneg = total_gneg / num_batches
        print(f"  Goodness - pos: {avg_gpos:.4f}, neg: {avg_gneg:.4f}")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation...")
    print("=" * 60)
    
    eval_config = EvaluationConfig(
        device=device,
        dims=(1, 2, 3),
        dims_in=(1, 2, 3),
        dims_out=(1, 2, 3),
        stdnorm_out=True,
        out_dropout=out_dropout,
        Layer_out=list(range(args.NL)),
        pre_std=True,
        all_neurons=False
    )
    
    loaders = (train_loader, test_loader, test_loader, sup_train_loader)
    acc_train, acc_test = evaluate_model(
        nets, pools, extra_pools, eval_config, loaders, search=False, Dims=Dims
    )
    
    print("\n" + "=" * 60)
    print("BASELINE RESULTS (10k samples)")
    print("=" * 60)
    print(f"Train Accuracy: {100 * acc_train:.2f}%")
    print(f"Test Accuracy:  {100 * acc_test:.2f}%")
    print("=" * 60)
    
    return acc_test


if __name__ == "__main__":
    main()
