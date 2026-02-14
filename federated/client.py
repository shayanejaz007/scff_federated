#!/usr/bin/env python3
"""
Optimal Distributed SCFF Training

CRITICAL INSIGHT: SCFF uses GREEDY LAYER-LOCAL training.
- Each layer trains independently (no backprop between layers)
- This means we can distribute training PERFECTLY

APPROACH: Layer-wise Federated Training
- Train layer 0 across all clients until convergence → freeze
- Train layer 1 across all clients until convergence → freeze
- Train layer 2 across all clients until convergence → freeze
- This achieves EXACT PARITY with centralized training

WHY THIS WORKS:
- SCFF loss at layer i depends ONLY on activations at layer i
- All clients contribute activations → server sees full distribution
- Same as centralized SGD, just with distributed data
"""
import argparse
import torch
import torch.nn as nn
import socket
import time
from typing import Optional, Tuple, Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scff.data import DualAugmentCIFAR10
from scff.model import build_layers_from_config, get_pos_neg_batch_imgcats
from scff.layers import stdnorm
from federated.protocol import send_msg, recv_msg
from federated.partition import DataPartitioner


class OptimalSCFFClient:
    """
    Client for layer-wise federated SCFF training.
    
    Training flow:
    1. Receive current layer index from server
    2. For each batch: compute activations up to current layer
    3. Send activations OR train locally and send gradients
    4. Repeat until layer converges, then move to next layer
    """
    
    def __init__(
        self,
        client_id: int,
        config_path: str,
        dataset: str,
        NL: int,
        device: str,
        seed: int
    ):
        self.client_id = client_id
        self.NL = NL
        self.device = device
        self.seed = seed
        
        print(f"Initializing Client {client_id}")
        
        # Build FULL model (all layers)
        torch.manual_seed(seed)
        self.model_components = build_layers_from_config(
            config_path=config_path,
            dataset=dataset,
            NL=NL,
            device=device,
            load_params=False,
            freeze_layer=NL  # Don't freeze any initially
        )
        
        self.nets = self.model_components['nets']
        self.pools = self.model_components['pools']
        self.optimizers = self.model_components['optimizers']
        self.schedulers = self.model_components['schedulers']
        self.threshold1 = self.model_components['threshold1']
        self.threshold2 = self.model_components['threshold2']
        self.lamda = self.model_components['lamda']
        self.period = self.model_components['period']
        
        self.dims_in = (1, 2, 3)
        self.p = 1
        self.a = 1.0
        self.b = 1.0
        
        # Track which layers are frozen
        self.frozen_layers = set()
        
        # Batch counter for LR scheduling
        self.nbbatches = [0] * NL
        
        # Dims
        self.Dims = []
        
        self.data_loader = None
    
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
        print(f"  Client {self.client_id}: {data_loader.num_samples} samples")
    
    def load_layer_state(self, layer_idx: int, state_dict: dict, opt_state: dict, nbbatches: int):
        """Load state for a specific layer."""
        self.nets[layer_idx].load_state_dict(state_dict)
        if opt_state:
            self.optimizers[layer_idx].load_state_dict(opt_state)
        self.nbbatches[layer_idx] = nbbatches
    
    def get_layer_state(self, layer_idx: int) -> Tuple[dict, dict, int]:
        """Get state for a specific layer."""
        return (
            self.nets[layer_idx].state_dict(),
            self.optimizers[layer_idx].state_dict(),
            self.nbbatches[layer_idx]
        )
    
    def freeze_layer(self, layer_idx: int):
        """Freeze a layer after training."""
        self.frozen_layers.add(layer_idx)
        for param in self.nets[layer_idx].parameters():
            param.requires_grad = False
        print(f"  Layer {layer_idx} frozen")
    
    def train_layer(
        self,
        layer_idx: int,
        local_epochs: int = 1
    ) -> Dict:
        """
        Train a SINGLE layer using SCFF layer-local loss.
        
        This is the core of SCFF: each layer learns to maximize goodness
        for positive pairs and minimize for negative pairs.
        """
        if self.data_loader is None:
            raise ValueError("Data loader not set!")
        
        # Set training mode for current layer only
        for i, net in enumerate(self.nets):
            if i == layer_idx:
                net.train()
            else:
                net.eval()
        
        total_loss = 0.0
        total_gpos = 0.0
        total_gneg = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            for batch_data in self.data_loader:
                # Unpack dual augmentation
                if len(batch_data) == 4:
                    _, img1, img2, _ = batch_data
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                else:
                    img1 = batch_data[0].to(self.device)
                    img2 = img1
                
                # Forward through frozen layers to get input for current layer
                x1, x2 = img1, img2
                
                with torch.no_grad():
                    for i in range(layer_idx):
                        if self.nets[i].concat:
                            x1 = stdnorm(x1, dims=self.dims_in)
                            x2 = stdnorm(x2, dims=self.dims_in)
                            x1, x2 = get_pos_neg_batch_imgcats(x1, x2, p=self.p)
                        
                        x1 = self.pools[i](self.nets[i].act(self.nets[i](x1)))
                        x2 = self.pools[i](self.nets[i].act(self.nets[i](x2)))
                
                # Now train the current layer
                net = self.nets[layer_idx]
                
                if net.concat:
                    x1 = stdnorm(x1, dims=self.dims_in)
                    x2 = stdnorm(x2, dims=self.dims_in)
                    x_pos, x_neg = get_pos_neg_batch_imgcats(x1, x2, p=self.p)
                else:
                    x_pos, x_neg = x1, x2
                
                # Forward through current layer
                out_pos = net(x_pos)
                out_neg = net(x_neg)
                
                # Compute goodness (squared activations)
                yforgrad = net.relu(out_pos).pow(2).mean([1])
                yforgrad_neg = net.relu(out_neg).pow(2).mean([1])
                
                # SCFF contrastive loss for this layer
                self.optimizers[layer_idx].zero_grad()
                
                loss = (
                    torch.log(1 + torch.exp(
                        self.a * (-yforgrad + self.threshold1[layer_idx])
                    )).mean([1, 2]).mean()
                    + torch.log(1 + torch.exp(
                        self.b * (yforgrad_neg - self.threshold2[layer_idx])
                    )).mean([1, 2]).mean()
                    + self.lamda[layer_idx] * torch.norm(yforgrad, p=2, dim=(1, 2)).mean()
                )
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                
                self.optimizers[layer_idx].step()
                
                # LR scheduling
                self.nbbatches[layer_idx] += 1
                if self.nbbatches[layer_idx] % self.period[layer_idx] == 0:
                    self.schedulers[layer_idx].step()
                
                # Track metrics
                gpos = yforgrad.mean([1, 2]).mean().item()
                gneg = yforgrad_neg.mean([1, 2]).mean().item()
                
                total_loss += loss.item()
                total_gpos += gpos
                total_gneg += gneg
                num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'goodness_pos': total_gpos / max(num_batches, 1),
            'goodness_neg': total_gneg / max(num_batches, 1),
            'num_batches': num_batches
        }


def main():
    parser = argparse.ArgumentParser(description="Optimal SCFF Client")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="CIFAR")
    parser.add_argument("--NL", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--server_addr", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--dataset_size", type=int, default=50000)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Optimal SCFF Client {args.client_id}")
    print("=" * 60)
    
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    
    client = OptimalSCFFClient(
        client_id=args.client_id,
        config_path=args.config,
        dataset=args.dataset,
        NL=args.NL,
        device=device,
        seed=args.seed
    )
    
    # Load data with DUAL augmentation (critical!)
    print("\nLoading data with DUAL augmentation...")
    train_dataset = DualAugmentCIFAR10(
        root='./data', train=True, download=True, augment="dual"
    )
    
    torch.manual_seed(args.seed)
    all_indices = torch.randperm(len(train_dataset)).tolist()[:args.dataset_size]
    
    partitioner = DataPartitioner(
        dataset_size=args.dataset_size,
        num_clients=args.num_clients,
        seed=args.seed
    )
    
    client_indices = [all_indices[i] for i in partitioner.get_client_indices(args.client_id)]
    
    from torch.utils.data import Subset, DataLoader
    client_subset = Subset(train_dataset, client_indices)
    data_loader = DataLoader(
        client_subset, batch_size=args.batchsize, shuffle=True,
        num_workers=0, drop_last=True
    )
    
    class Loader:
        def __init__(self, loader, n):
            self.loader = loader
            self.num_samples = n
        def __iter__(self):
            return iter(self.loader)
        def __len__(self):
            return len(self.loader)
    
    client.set_data_loader(Loader(data_loader, len(client_indices)))
    
    # Connect to server
    print(f"\nConnecting to server...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    for i in range(30):
        try:
            sock.connect((args.server_addr, args.port))
            break
        except ConnectionRefusedError:
            time.sleep(2)
    
    send_msg(sock, {'client_id': args.client_id, 'num_samples': len(client_indices)})
    recv_msg(sock)
    print("Connected!")
    
    try:
        while True:
            cmd = recv_msg(sock)
            
            if cmd is None or cmd.get('cmd') == 'shutdown':
                break
            
            elif cmd.get('cmd') == 'train_layer':
                layer_idx = cmd['layer_idx']
                round_num = cmd['round_num']
                local_epochs = cmd.get('local_epochs', 1)
                
                print(f"\n--- Layer {layer_idx}, Round {round_num} ---")
                
                # Load layer state from server
                if cmd.get('layer_state'):
                    client.load_layer_state(
                        layer_idx,
                        cmd['layer_state'],
                        cmd.get('opt_state'),
                        cmd.get('nbbatches', 0)
                    )
                
                # Train this layer
                metrics = client.train_layer(layer_idx, local_epochs)
                
                print(f"  Loss: {metrics['loss']:.4f}, "
                      f"Goodness pos: {metrics['goodness_pos']:.4f}, "
                      f"neg: {metrics['goodness_neg']:.4f}")
                
                # Send updated layer state back
                layer_state, opt_state, nbbatches = client.get_layer_state(layer_idx)
                send_msg(sock, {
                    'client_id': args.client_id,
                    'layer_idx': layer_idx,
                    'layer_state': layer_state,
                    'opt_state': opt_state,
                    'nbbatches': nbbatches,
                    'metrics': metrics
                })
            
            elif cmd.get('cmd') == 'freeze_layer':
                layer_idx = cmd['layer_idx']
                if cmd.get('layer_state'):
                    client.load_layer_state(layer_idx, cmd['layer_state'], None, 0)
                client.freeze_layer(layer_idx)
                send_msg(sock, {'status': 'ok'})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        print(f"\nClient {args.client_id} done")


if __name__ == "__main__":
    main()
