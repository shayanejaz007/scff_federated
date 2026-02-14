"""
SCFF Evaluation
Exact implementation matching original SCFF_CIFAR.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .layers import stdnorm


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    device: str
    dims: Tuple[int, ...] = (1, 2, 3)
    dims_in: Tuple[int, ...] = (1, 2, 3)
    dims_out: Tuple[int, ...] = (1, 2, 3)
    stdnorm_out: bool = True
    out_dropout: float = 0.2  # Original SCFF value
    Layer_out: List[int] = None
    pre_std: bool = True
    all_neurons: bool = False
    readout_epochs: int = 50  # Original SCFF value
    weight_decay: float = 0.0  # Original SCFF has no weight decay
    
    def __post_init__(self):
        if self.Layer_out is None:
            self.Layer_out = [0, 1, 2]


class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """
    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5 for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


def calculate_output_length(dims: List[int], nets: List[nn.Module], extra_pool: List[nn.Module], 
                           Layer: List[int], all_neurons: bool) -> int:
    """
    Calculate the total output feature length for the classifier.
    
    Args:
        dims: List of dimension sizes for each layer
        nets: List of network layers
        extra_pool: List of extra pooling layers
        Layer: List of layer indices to include in output
        all_neurons: Whether to use all neurons or apply extra pooling
        
    Returns:
        Total feature length
    """
    lengths = 0
    if all_neurons:
        for i, length in enumerate(dims):
            if i in Layer:
                lengths += length
    else:
        for i, length in enumerate(dims):
            if i in Layer:
                len_after_pool = math.ceil(
                    (math.sqrt(length / nets[i].output_channels) - extra_pool[i].kernel_size) 
                    / extra_pool[i].stride + 1
                )
                lengths += len_after_pool * len_after_pool * nets[i].output_channels
    return lengths


def build_classifier(lengths: int, config: EvaluationConfig) -> nn.Module:
    """
    Build the linear classifier head.
    
    Args:
        lengths: Input feature dimension
        config: Evaluation configuration
        
    Returns:
        Classifier module
    """
    classifier = nn.Sequential(
        nn.Dropout(config.out_dropout),
        nn.Linear(lengths, 10)  # CIFAR-10 has 10 classes
    ).to(config.device)
    
    if torch.cuda.device_count() > 2:
        classifier = nn.DataParallel(classifier)
    
    return classifier


def train_readout(classifier: nn.Module, nets: List[nn.Module], pool: List[nn.Module], 
                  extra_pool: List[nn.Module], loader, criterion, optimizer, 
                  config: EvaluationConfig, epoch: int) -> float:
    """
    Train the readout classifier for one epoch.
    
    Args:
        classifier: The linear classifier
        nets: List of feature extractor layers
        pool: List of pooling layers
        extra_pool: List of extra pooling layers
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        config: Evaluation configuration
        epoch: Current epoch number
        
    Returns:
        Training accuracy
    """
    classifier.train()
    correct = 0
    total = 0

    for i, (x, labels) in enumerate(loader):
        x = x.to(config.device)
        labels = labels.to(config.device)

        outputs = []
        
        with torch.no_grad():
            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm(x, dims=config.dims_in)
                    x = torch.cat((x, x), dim=1)

                x = pool[j](net.act(net(x)))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm(out, dims=config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

        outputs = torch.cat(outputs, dim=1)
        optimizer.zero_grad()
        outputs = classifier(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return correct / total


def test_readout(classifier: nn.Module, nets: List[nn.Module], pool: List[nn.Module],
                 extra_pool: List[nn.Module], loader, criterion, 
                 config: EvaluationConfig, epoch: int, mode: str) -> float:
    """
    Test the readout classifier.
    
    Args:
        classifier: The linear classifier
        nets: List of feature extractor layers
        pool: List of pooling layers
        extra_pool: List of extra pooling layers
        loader: Data loader
        criterion: Loss function
        config: Evaluation configuration
        epoch: Current epoch number
        mode: 'Train' or 'Val' for logging
        
    Returns:
        Test accuracy
    """
    classifier.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (x, labels) in enumerate(loader):
            x = x.to(config.device)
            labels = labels.to(config.device)
            outputs = []
            
            for j, net in enumerate(nets):
                if net.concat:
                    x = stdnorm(x, dims=config.dims_in)
                    x = torch.cat((x, x), dim=1)
                    
                x = pool[j](net.act(net(x)))

                if not config.all_neurons:
                    out = extra_pool[j](x)

                if config.stdnorm_out:
                    out = stdnorm(out, dims=config.dims_out)
                out = out.flatten(start_dim=1)
                if j in config.Layer_out:
                    outputs.append(out)

            outputs = torch.cat(outputs, dim=1)
            outputs = classifier(outputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    if mode == 'Val':
        print(f'Accuracy of the network on the 10000 {mode} images: {100 * correct / total} %')
        print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

    return correct / total


def evaluate_model(nets: List[nn.Module], pool: List[nn.Module], extra_pool: List[nn.Module],
                   config: EvaluationConfig, loaders: Tuple, search: bool, 
                   Dims: List[int]) -> Tuple[float, float]:
    """
    Evaluates a trained neural network model by training a classifier on top of extracted features.

    Args:
        nets: List of trained neural network layers.
        pool: List of pooling layers corresponding to each network layer.
        extra_pool: Additional pooling layers for feature extraction.
        config: Configuration containing evaluation parameters.
        loaders: Tuple containing (trainloader, valloader, testloader, suptrloader).
        search: If True, validation is done using the test dataset.
        Dims: Dimensions of each layer's output feature map.

    Returns:
        Tuple of (acc_train, acc_val)
    """
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    
    lengths = calculate_output_length(Dims, nets, extra_pool, config.Layer_out, config.all_neurons)
    print(lengths)
    classifier = build_classifier(lengths, config)
    
    _, valloader, testloader, suptrloader = loaders
    
    # Optimizer with weight decay to prevent overfitting
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=config.weight_decay)
    lr_scheduler = CustomStepLR(optimizer, nb_epochs=config.readout_epochs)
    criterion = nn.CrossEntropyLoss()

    if not search:
        valloader = testloader
    
    # Set all feature extractors to eval mode
    for j, net in enumerate(nets):
        net.eval()

    # Main evaluation loop with configurable epochs
    for epoch in range(config.readout_epochs):
        acc_train = train_readout(classifier, nets, pool, extra_pool, suptrloader, 
                                  criterion, optimizer, config, epoch)
        lr_scheduler.step()
        if epoch % 10 == 0 or epoch == config.readout_epochs - 1:
            print(f'Accuracy of the network on the 50000 train images: {100 * acc_train} %')
            acc_train = test_readout(classifier, nets, pool, extra_pool, suptrloader,
                                     criterion, config, epoch, 'Train')
            acc_val = test_readout(classifier, nets, pool, extra_pool, valloader,
                                   criterion, config, epoch, 'Val')

    torch.set_rng_state(current_rng_state)
    
    return acc_train, acc_val
