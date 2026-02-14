"""
SCFF Training Loop
Exact implementation matching original SCFF_CIFAR.py
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .layers import stdnorm
from .model import get_pos_neg_batch_imgcats
from .eval import evaluate_model


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    device: str
    dims: Tuple[int, ...] = (1, 2, 3)
    dims_in: Tuple[int, ...] = (1, 2, 3)
    dims_out: Tuple[int, ...] = (1, 2, 3)
    stdnorm_out: bool = True
    out_dropout: float = 0.2
    Layer_out: List[int] = None
    pre_std: bool = True
    all_neurons: bool = False
    
    def __post_init__(self):
        if self.Layer_out is None:
            self.Layer_out = [0, 1, 2]


class SCFFTrainer:
    """
    SCFF Trainer implementing greedy layer-local training.
    
    This trainer implements the exact training loop from the original SCFF code,
    with support for layer-local (no end-to-end backprop) training.
    """
    
    def __init__(
        self,
        nets: List[nn.Module],
        pools: List[nn.Module],
        extra_pools: List[nn.Module],
        optimizers: List,
        schedulers: List,
        threshold1: List[float],
        threshold2: List[float],
        lamda: List[float],
        period: List[int],
        device: str,
        dims_in: Tuple[int, ...] = (1, 2, 3),
        dims_out: Tuple[int, ...] = (1, 2, 3),
        a: float = 1.0,
        b: float = 1.0,
        p: int = 1,
    ):
        self.nets = nets
        self.pools = pools
        self.extra_pools = extra_pools
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.lamda = lamda
        self.period = period
        self.device = device
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.a = a
        self.b = b
        self.p = p
        self.NL = len(nets)
        
    def train_epoch(
        self,
        trainloader,
        epoch: int,
        freeze_layer: int,
        nbbatches: int,
        firstpass: bool = False,
        Dims: Optional[List[int]] = None
    ) -> Tuple[int, bool, List[int], float, float]:
        """
        Train one epoch using greedy layer-local updates.
        
        Args:
            trainloader: Training data loader
            epoch: Current epoch number
            freeze_layer: Layers below this index are frozen
            nbbatches: Running count of batches processed
            firstpass: Whether this is the first pass (for dimension logging)
            Dims: List to store layer output dimensions
            
        Returns:
            Tuple of (nbbatches, firstpass, Dims, goodness_pos_avg, goodness_neg_avg)
        """
        if Dims is None:
            Dims = []
            
        # Set training mode for last layer only
        self.nets[-1].train()
        
        goodness_pos = 0
        goodness_neg = 0
        
        for numbatch, (x, _) in enumerate(trainloader):
            nbbatches += 1
            x = x.to(self.device)
            
            # Process through all layers
            for i in range(self.NL):
                # Build pos/neg batch if this layer uses concat
                if self.nets[i].concat:
                    x = stdnorm(x, dims=self.dims_in)
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=self.p)
                
                # Forward pass
                x = self.nets[i](x)
                x_neg = self.nets[i](x_neg)
                
                # Compute goodness (squared activations)
                yforgrad = self.nets[i].relu(x).pow(2).mean([1])
                yforgrad_neg = self.nets[i].relu(x_neg).pow(2).mean([1])
                
                # Determine if this layer should be trained
                if i < freeze_layer:
                    UNLAB = False
                else:
                    UNLAB = True
                
                # Layer-local update (no end-to-end backprop)
                if UNLAB:
                    self.optimizers[i].zero_grad()
                    
                    # SCFF loss: push positive goodness up, negative goodness down
                    loss = (
                        torch.log(1 + torch.exp(
                            self.a * (-yforgrad + self.threshold1[i])
                        )).mean([1, 2]).mean() 
                        + torch.log(1 + torch.exp(
                            self.b * (yforgrad_neg - self.threshold2[i])
                        )).mean([1, 2]).mean() 
                        + self.lamda[i] * torch.norm(yforgrad, p=2, dim=(1, 2)).mean()
                    )
                    
                    # Layer-local backward pass
                    loss.backward()
                    self.optimizers[i].step()
                    
                    # Learning rate scheduling
                    if (nbbatches + 1) % self.period[i] == 0:
                        self.schedulers[i].step()
                        print(f'nbbatches {nbbatches + 1} learning rate: {self.schedulers[i].get_last_lr()[0]}')
                
                # CRITICAL: Detach before passing to next layer (greedy training)
                x = self.pools[i](self.nets[i].act(x)).detach()
                x_neg = self.pools[i](self.nets[i].act(x_neg)).detach()
                
                if firstpass:
                    print(f"Layer {i}: x.shape: {x.shape} y.shape (after MaxP): {x.shape}", end=" ")
                    _, channel, h, w = x.shape
                    Dims.append(channel * h * w)
            
            firstpass = False
            goodness_pos += torch.mean(yforgrad.mean([1, 2])).item()
            goodness_neg += torch.mean(yforgrad_neg.mean([1, 2])).item()
            
            # Print epoch statistics at end of epoch
            if UNLAB and numbatch == len(trainloader) - 1:
                print(goodness_pos / len(trainloader), goodness_neg / len(trainloader))
        
        return nbbatches, firstpass, Dims, goodness_pos / len(trainloader), goodness_neg / len(trainloader)
    
    def train(
        self,
        trainloader,
        valloader,
        testloader,
        suptrloader,
        epochs: int,
        freeze_layer: int,
        tr_and_eval: bool = False,
        eval_config: Optional[EvaluationConfig] = None,
        search: bool = False
    ) -> Tuple[List[List[float]], List[List[float]], List[int], Optional[List[float]]]:
        """
        Full training loop.
        
        Args:
            trainloader: Training data loader
            valloader: Validation data loader
            testloader: Test data loader
            suptrloader: Supervised training data loader for evaluation
            epochs: Number of training epochs
            freeze_layer: Layers below this index are frozen
            tr_and_eval: Whether to evaluate during training
            eval_config: Configuration for evaluation
            search: Whether to use test set as validation
            
        Returns:
            Tuple of (all_pos, all_neg, Dims, taccs)
        """
        all_pos = [[] for _ in range(self.NL)]
        all_neg = [[] for _ in range(self.NL)]
        
        firstpass = True
        nbbatches = 0
        NBLEARNINGEPOCHS = epochs
        
        if epochs == 0:
            N_all = NBLEARNINGEPOCHS + 1
        else:
            N_all = NBLEARNINGEPOCHS
        
        Dims = []
        taccs = []
        
        for epoch in range(N_all):
            print(f"Epoch {epoch}")
            
            if epoch < NBLEARNINGEPOCHS and epochs != 0:
                self.nets[-1].train()
                print("Unlabeled.")
                zeloader = trainloader
                
                nbbatches, firstpass, Dims, gpos, gneg = self.train_epoch(
                    zeloader, epoch, freeze_layer, nbbatches, firstpass, Dims
                )
                
                all_pos[-1].append(gpos * len(zeloader))
                all_neg[-1].append(gneg * len(zeloader))
                
            else:
                # Evaluate the output neurons without training
                print("Evaluate the trained features.")
                for net in self.nets:
                    net.eval()
                
                if epoch == NBLEARNINGEPOCHS:
                    zeloader = testloader
                else:
                    raise ValueError("Wrong epoch!")
            
            # Periodic evaluation during training
            if tr_and_eval and epoch > 0 and epoch % 1 == 0:
                loaders = (trainloader, valloader, testloader, suptrloader)
                tacc = evaluate_model(
                    self.nets, self.pools, self.extra_pools, 
                    eval_config, loaders, search, Dims
                )
                taccs.append(tacc)
        
        print("Training done..")
        
        return all_pos, all_neg, Dims, taccs if tr_and_eval else None


def train(
    nets, device, optimizers, schedulers, threshold1, threshold2, dims_in, dims_out,
    epochs, pool, a, b, lamda, freezelayer, period, extra_pool, tr_and_eval, Layer_out,
    trainloader, valloader, testloader, suptrloader, p, config
):
    """
    Legacy training function matching original SCFF_CIFAR.py signature.
    
    This function provides backwards compatibility with the original code.
    """
    all_pos = [[] for _ in range(len(nets))]
    all_neg = [[] for _ in range(len(nets))]
    
    NL = len(nets)
    firstpass = True
    nbbatches = 0
    NBLEARNINGEPOCHS = epochs

    if epochs == 0:
        N_all = NBLEARNINGEPOCHS + 1
    else:
        N_all = NBLEARNINGEPOCHS

    Dims = []
    taccs = []
    
    # Start the experiment
    for epoch in range(N_all):
        print(f"Epoch {epoch}")
        if epoch < NBLEARNINGEPOCHS and epochs != 0:
            nets[-1].train()
            print("Unlabeled.")
            UNLAB = True
            zeloader = trainloader
        else:
            print("Evaluate the trained features.")
            for net in nets:
                net.eval()
            if epoch == NBLEARNINGEPOCHS:
                UNLAB = False
                zeloader = testloader
            else:
                raise ValueError("Wrong epoch!")

        goodness_pos = 0
        goodness_neg = 0

        for numbatch, (x, _) in enumerate(zeloader):
            nbbatches += 1
            x = x.to(device)

            for i in range(NL):
                if nets[i].concat:
                    x = stdnorm(x, dims=dims_in)
                    x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)

                x = nets[i](x)
                x_neg = nets[i](x_neg)

                yforgrad = nets[i].relu(x).pow(2).mean([1])
                yforgrad_neg = nets[i].relu(x_neg).pow(2).mean([1])

                if i < freezelayer:
                    UNLAB = False
                else:
                    UNLAB = True

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

                    if (nbbatches + 1) % period[i] == 0:
                        schedulers[i].step()
                        print(f'nbbatches {nbbatches + 1} learning rate: {schedulers[i].get_last_lr()[0]}')

                x = pool[i](nets[i].act(x)).detach()
                x_neg = pool[i](nets[i].act(x_neg)).detach()

                if firstpass:
                    print(f"Layer {i}: x.shape: {x.shape} y.shape (after MaxP): {x.shape}", end=" ")
                    _, channel, h, w = x.shape
                    Dims.append(channel * h * w)

            firstpass = False
            goodness_pos += torch.mean(yforgrad.mean([1, 2])).item()
            goodness_neg += torch.mean(yforgrad_neg.mean([1, 2])).item()

            if UNLAB and numbatch == len(zeloader) - 1:
                print(goodness_pos / len(zeloader), goodness_neg / len(zeloader))
                all_pos[i].append(goodness_pos)
                all_neg[i].append(goodness_neg)
                goodness_pos, goodness_neg = 0, 0

        if tr_and_eval:
            if epoch > 0 and epoch % 1 == 0:
                loaders = (trainloader, valloader, testloader, suptrloader)
                tacc = evaluate_model(nets, pool, extra_pool, config, loaders, False, Dims)
                taccs.append(tacc)

    print("Training done..")
    
    if tr_and_eval:
        return nets, all_pos, all_neg, Dims, taccs
    else:
        return nets, all_pos, all_neg, Dims
