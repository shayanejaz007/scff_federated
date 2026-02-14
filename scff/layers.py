"""
SCFF Layer Definitions
Exact implementation matching original SCFF_CIFAR.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def stdnorm(x, dims=(1, 2, 3)):
    """
    Standard normalization (zero mean, unit variance).
    
    Args:
        x: Input tensor
        dims: Dimensions to normalize over
        
    Returns:
        Normalized tensor
    """
    x = x - torch.mean(x, dim=dims, keepdim=True)
    x = x / (1e-10 + torch.std(x, dim=dims, keepdim=True))
    return x


class standardnorm(nn.Module):
    """Standard normalization layer (zero mean, unit variance)."""
    
    def __init__(self, dims=(1, 2, 3)):
        super(standardnorm, self).__init__()
        self.dims = dims

    def forward(self, x):
        x = x - torch.mean(x, dim=self.dims, keepdim=True)
        x = x / (1e-10 + torch.std(x, dim=self.dims, keepdim=True))
        return x


class L2norm(nn.Module):
    """L2 normalization layer."""
    
    def __init__(self, dims=(1, 2, 3)):
        super(L2norm, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x / (x.norm(p=2, dim=self.dims, keepdim=True) + 1e-10)


class triangle(nn.Module):
    """Triangle activation: mean-centered ReLU."""
    
    def __init__(self):
        super(triangle, self).__init__()

    def forward(self, x):
        x = x - torch.mean(x, axis=1, keepdims=True)
        return F.relu(x)


class Conv2d(nn.Module):
    """
    A custom 2D convolutional layer with optional normalization/standardization, activation, 
    and concatenation of input channels for self-contrastive inputs.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        kernel_size (tuple or int): Size of the convolutional kernel.
        pad (int, optional): Padding size.
        batchnorm (bool, optional): Whether to apply batch normalization. Default is False.
        normdims (list, optional): Dimensions to apply normalization over. Default is [1,2,3].
        norm (str, optional): Normalization type, 'stdnorm' for standard normalization or 'L2norm'. Default is 'stdnorm'.
        bias (bool, optional): Whether to use bias in convolution. Default is True.
        dropout (float, optional): Dropout rate. Default is 0.0.
        padding_mode (str, optional): Padding mode for convolution (e.g., 'zeros', 'reflect'). Default is 'reflect'.
        concat (bool, optional): Whether to split input channels and apply convolution separately before summing. Default is True.
        act (str, optional): Activation function for transmitting information to the next layer not for plastisity, 'relu' or 'triangle'. Default is 'relu'.
    """
    
    def __init__(
        self, 
        input_channels, 
        output_channels, 
        kernel_size, 
        pad=0, 
        batchnorm=False, 
        normdims=(1, 2, 3), 
        norm="stdnorm",
        bias=True, 
        dropout=0.0, 
        padding_mode="reflect", 
        concat=True, 
        act="relu"
    ):
        super(Conv2d, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.normdims = normdims
        self.concat = concat  # If True, input channels are split and processed separately because of concatenated pos/neg images
        self.relu = torch.nn.ReLU()

        # Define convolutional layer
        self.conv_layer = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=kernel_size, 
            bias=bias
        )
        
        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(self.conv_layer.weight)
        
        # Set padding parameters
        self.padding_mode = padding_mode
        self.F_padding = (pad, pad, pad, pad)  # Symmetric padding on all sides
        
        # Define activation function
        if act == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = triangle()
        
        # Apply batch normalization if enabled
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.input_channels, affine=False)
        else:
            self.bn1 = nn.Identity()

        # Select normalization type (Standard Normalization or L2 Normalization)
        if norm == "L2norm":
            self.norm = L2norm(dims=normdims)
        elif norm == "stdnorm":
            self.norm = standardnorm(dims=normdims)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after convolution without activation.
        """
        # batchnorm is false by default
        x = self.bn1(x) 
        x = F.pad(x, self.F_padding, self.padding_mode)
        x = self.norm(x)  # standardization before convolutions

        if self.concat: 
            lenchannel = x.size(1) // 2
            # If concat mode is enabled, split channels into two halves, apply convolution separately, and sum results
            out = self.conv_layer(x[:, :lenchannel]) + self.conv_layer(x[:, lenchannel:])
        else:
            out = self.conv_layer(x)
        
        return out
