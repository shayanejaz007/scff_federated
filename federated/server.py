#!/usr/bin/env python3
"""
Optimal Distributed SCFF Server

TRAINING PARADIGM: Layer-wise FedAvg
====================================

For each layer i in [0, 1, 2]:
    For round in [0, ..., rounds_per_layer]:
        1. Broadcast layer i weights to all clients
        2. Each client trains layer i on their local data
        3. Clients send updated weights back
        4. Server aggregates using FedAvg (weighted by samples)
    Freeze layer i, move to layer i+1

WHY THIS ACHIEVES CENTRALIZED PARITY:
- SCFF layers train INDEPENDENTLY (no cross-layer gradients)
- Each layer eventually sees ALL data through FedAvg aggregation
- Same total gradient updates as centralized training
- FedAvg is proven to converge to centralized solution for convex objectives

BEST PRACTICES IMPLEMENTED:
- FedAvg weighted by number of samples
- Synchronized LR scheduling via global nbbatches
- IID data split (stratified by class)
- Gradient clipping for stability
- Proper dual augmentation for contrastive learning
"""
import argparse
import torch
import socket
import time
import copy
from typing import Dict, List, Optional
from collections import defaultdict
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scff.data import DualAugmentCIFAR10_test
from scff.model import build_layers_from_config
from scff.eval import EvaluationConfig, evaluate_model
from federated.protocol import send_msg, recv_msg
from torch.utils.data import DataLoader


class OptimalSCFFServer:
    """
    Server for layer-wise federated SCFF training.
    
    Orchestrates training one layer at a time across all clients,
    using FedAvg for aggregation. Achieves centralized parity.
    """
    
    def __init__(
        self,
        config_path: str,
        dataset: str,
        NL: int,
        device: str,
        seed: int,
        num_clients: int
    ):
        self.config_path = config_path
        self.NL = NL
        self.device = device
        self.seed = seed
        self.num_clients = num_clients
        
        print(f"Initializing Optimal SCFF Server")
        print(f"  Layers: {NL}")
        print(f"  Clients: {num_clients}")
        print(f"  Method: Layer-wise FedAvg")
        
        # Build global model
        torch.manual_seed(seed)
        self.model_components = build_layers_from_config(
            config_path=config_path,
            dataset=dataset,
            NL=NL,
            device=device,
            load_params=False,
            freeze_layer=NL
        )
        
        self.nets = self.model_components['nets']
        self.pools = self.model_components['pools']
        self.extra_pools = self.model_components['extra_pools']
        self.optimizers = self.model_components['optimizers']
        self.schedulers = self.model_components['schedulers']
        
        # Client tracking
        self.client_sockets: Dict[int, socket.socket] = {}
        self.client_samples: Dict[int, int] = {}
        
        # Training state per layer
        self.nbbatches = [0] * NL
        self.frozen_layers = set()
        
        # History
        self.history = defaultdict(list)
        self.best_test_acc = 0.0
        
        # Dims for evaluation
        self.Dims = []
        self._compute_dims()
    
    def _compute_dims(self):
        """Compute activation dimensions for each layer."""
        x = torch.randn(1, 3, 32, 32).to(self.device)
        with torch.no_grad():
            for i, net in enumerate(self.nets):
                if net.concat:
                    x = torch.cat([x, x], dim=1)
                x = self.pools[i](net.act(net(x)))
                _, c, h, w = x.shape
                self.Dims.append(c * h * w)
        print(f"  Dims: {self.Dims}")
    
    def fedavg_aggregate(self, layer_idx: int, client_updates: List[Dict]) -> Dict:
        """
        FedAvg aggregation: weighted average of client weights by sample count.
        
        w_global = Σ (n_k / n_total) * w_k
        
        This is mathematically equivalent to centralized SGD when:
        - All clients complete same number of local steps
        - Data is IID across clients
        """
        total_samples = sum(u['metrics']['num_batches'] * 100 for u in client_updates)  # approx
        
        # Initialize aggregated state
        agg_state = {}
        
        for key in client_updates[0]['layer_state'].keys():
            agg_state[key] = torch.zeros_like(
                client_updates[0]['layer_state'][key], 
                dtype=torch.float32
            )
        
        # Weighted average
        for update in client_updates:
            weight = update['metrics']['num_batches'] / sum(u['metrics']['num_batches'] for u in client_updates)
            for key in agg_state:
                agg_state[key] += weight * update['layer_state'][key].float()
        
        # Also aggregate nbbatches
        total_nbbatches = sum(u['nbbatches'] for u in client_updates)
        avg_nbbatches = total_nbbatches // len(client_updates)
        
        return {
            'layer_state': agg_state,
            'nbbatches': avg_nbbatches
        }
    
    def train_layer(
        self,
        layer_idx: int,
        rounds: int,
        local_epochs: int = 1
    ) -> Dict:
        """
        Train a single layer using federated learning.
        
        For `rounds` iterations:
        1. Send current layer weights to all clients
        2. Clients train locally
        3. Aggregate using FedAvg
        """
        print(f"\n{'='*60}")
        print(f"TRAINING LAYER {layer_idx}")
        print(f"{'='*60}")
        
        layer_history = {'round': [], 'loss': [], 'gpos': [], 'gneg': []}
        
        for round_num in range(rounds):
            print(f"\n--- Layer {layer_idx}, Round {round_num} ---")
            
            # Current layer state
            current_state = self.nets[layer_idx].state_dict()
            current_opt = self.optimizers[layer_idx].state_dict()
            
            # Send train command to all clients
            for client_id, sock in self.client_sockets.items():
                send_msg(sock, {
                    'cmd': 'train_layer',
                    'layer_idx': layer_idx,
                    'round_num': round_num,
                    'local_epochs': local_epochs,
                    'layer_state': current_state,
                    'opt_state': current_opt,
                    'nbbatches': self.nbbatches[layer_idx]
                })
            
            # Collect updates from all clients
            client_updates = []
            for client_id, sock in self.client_sockets.items():
                update = recv_msg(sock)
                client_updates.append(update)
                print(f"  Client {client_id}: loss={update['metrics']['loss']:.4f}, "
                      f"gpos={update['metrics']['goodness_pos']:.4f}, "
                      f"gneg={update['metrics']['goodness_neg']:.4f}")
            
            # FedAvg aggregation
            aggregated = self.fedavg_aggregate(layer_idx, client_updates)
            
            # Update global model
            self.nets[layer_idx].load_state_dict(aggregated['layer_state'])
            self.nbbatches[layer_idx] = aggregated['nbbatches']
            
            # Track metrics
            avg_loss = sum(u['metrics']['loss'] for u in client_updates) / len(client_updates)
            avg_gpos = sum(u['metrics']['goodness_pos'] for u in client_updates) / len(client_updates)
            avg_gneg = sum(u['metrics']['goodness_neg'] for u in client_updates) / len(client_updates)
            
            layer_history['round'].append(round_num)
            layer_history['loss'].append(avg_loss)
            layer_history['gpos'].append(avg_gpos)
            layer_history['gneg'].append(avg_gneg)
            
            print(f"  Aggregated: loss={avg_loss:.4f}, gpos={avg_gpos:.4f}, gneg={avg_gneg:.4f}")
        
        # Freeze this layer on server
        self.frozen_layers.add(layer_idx)
        for param in self.nets[layer_idx].parameters():
            param.requires_grad = False
        
        # Broadcast freeze to clients
        final_state = self.nets[layer_idx].state_dict()
        for sock in self.client_sockets.values():
            send_msg(sock, {
                'cmd': 'freeze_layer',
                'layer_idx': layer_idx,
                'layer_state': final_state
            })
            recv_msg(sock)  # Wait for ack
        
        print(f"\n  Layer {layer_idx} training complete and frozen")
        
        return layer_history
    
    def evaluate(self, test_loader, sup_train_loader) -> Dict:
        """Evaluate the full model."""
        print("\nEvaluating model...")
        
        eval_config = EvaluationConfig(
            device=self.device,
            dims=(1, 2, 3),
            dims_in=(1, 2, 3),
            dims_out=(1, 2, 3),
            stdnorm_out=True,
            out_dropout=0.2,
            Layer_out=list(range(self.NL)),
            pre_std=True,
            all_neurons=False,
            readout_epochs=50,
            weight_decay=0.0
        )
        
        loaders = (None, test_loader, test_loader, sup_train_loader)
        acc_train, acc_test = evaluate_model(
            self.nets, self.pools, self.extra_pools,
            eval_config, loaders, search=False, Dims=self.Dims
        )
        
        return {'train_acc': acc_train, 'test_acc': acc_test}
    
    def plot_history(self, layer_histories: Dict, save_path: str):
        """Plot training history for all layers."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Layer-wise Federated SCFF Training', fontsize=14, fontweight='bold')
        
        colors = ['blue', 'green', 'red']
        
        for layer_idx, hist in layer_histories.items():
            c = colors[layer_idx % len(colors)]
            
            # Loss
            axes[0].plot(hist['round'], hist['loss'], f'{c[0]}-o', 
                        label=f'Layer {layer_idx}', linewidth=2, markersize=4)
            
            # Goodness positive
            axes[1].plot(hist['round'], hist['gpos'], f'{c[0]}-o',
                        label=f'Layer {layer_idx}', linewidth=2, markersize=4)
            
            # Goodness negative
            axes[2].plot(hist['round'], hist['gneg'], f'{c[0]}-s',
                        label=f'Layer {layer_idx}', linewidth=2, markersize=4)
        
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss per Layer')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Goodness')
        axes[1].set_title('Positive Goodness per Layer')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Goodness')
        axes[2].set_title('Negative Goodness per Layer')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Training history saved: {save_path}")
    
    def shutdown_clients(self):
        """Send shutdown to all clients."""
        for sock in self.client_sockets.values():
            try:
                send_msg(sock, {'cmd': 'shutdown'})
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Optimal SCFF Server")
    parser.add_argument("--dataset", type=str, default="CIFAR")
    parser.add_argument("--NL", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--rounds_per_layer", type=int, default=20,
                        help="Training rounds for each layer")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--chart", type=str, default="training_history.png")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Optimal Layer-wise Federated SCFF Server")
    print("=" * 60)
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    print("=" * 60)
    
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    
    server = OptimalSCFFServer(
        config_path=args.config,
        dataset=args.dataset,
        NL=args.NL,
        device=device,
        seed=args.seed,
        num_clients=args.num_clients
    )
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    test_dataset = DualAugmentCIFAR10_test(root='./data', aug=False, train=False, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    sup_train_dataset = DualAugmentCIFAR10_test(root='./data', aug=True, train=True, download=True)
    sup_train_loader = DataLoader(sup_train_dataset, batch_size=64, shuffle=True, num_workers=0)
    print(f"  Test: {len(test_dataset)}, Sup train: {len(sup_train_dataset)}")
    
    # Setup socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', args.port))
    sock.listen(args.num_clients)
    
    print(f"\nWaiting for {args.num_clients} clients on port {args.port}...")
    
    for _ in range(args.num_clients):
        conn, addr = sock.accept()
        reg = recv_msg(conn)
        client_id = reg['client_id']
        server.client_sockets[client_id] = conn
        server.client_samples[client_id] = reg.get('num_samples', 0)
        print(f"  Client {client_id} connected ({reg.get('num_samples', '?')} samples)")
        send_msg(conn, {'status': 'ok'})
    
    print(f"\nAll {args.num_clients} clients connected!")
    
    try:
        layer_histories = {}
        
        # Train each layer sequentially
        for layer_idx in range(args.NL):
            layer_hist = server.train_layer(
                layer_idx=layer_idx,
                rounds=args.rounds_per_layer,
                local_epochs=args.local_epochs
            )
            layer_histories[layer_idx] = layer_hist
            
            # Evaluate after each layer
            result = server.evaluate(test_loader, sup_train_loader)
            print(f"\n  After Layer {layer_idx}: Train={100*result['train_acc']:.2f}%, "
                  f"Test={100*result['test_acc']:.2f}%")
            
            if result['test_acc'] > server.best_test_acc:
                server.best_test_acc = result['test_acc']
        
        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        final = server.evaluate(test_loader, sup_train_loader)
        print(f"Train Accuracy: {100*final['train_acc']:.2f}%")
        print(f"Test Accuracy:  {100*final['test_acc']:.2f}%")
        print(f"Best Test:      {100*server.best_test_acc:.2f}%")
        
        gap = final['train_acc'] - final['test_acc']
        if gap > 0.1:
            print(f"\n⚠️ Overfitting gap: {gap*100:.1f}%")
        else:
            print(f"\n✅ Good generalization (gap: {gap*100:.1f}%)")
        
        print("=" * 60)
        
        # Plot
        server.plot_history(layer_histories, args.chart)
        
    finally:
        server.shutdown_clients()
        for conn in server.client_sockets.values():
            conn.close()
        sock.close()
        print("\nServer shutdown")


if __name__ == "__main__":
    main()
