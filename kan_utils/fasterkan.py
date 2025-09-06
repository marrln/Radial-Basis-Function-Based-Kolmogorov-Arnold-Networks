"""
FasterKAN: Radial Basis Function-based Kolmogorov-Arnold Networks

This module implements a version of RBF-KANs of Delis, called FasterKAN (Kolmogorov-Arnold Networks using 
Radial Basis Functions) with modifications for better training and hardware implementation.

Modifications compared to the original FasterKAN implementation:
- Dropout with scaling based on the number of grids
- Linear layers without bias for FPGA compatibility
- Gradient scaling for grid and inverse denominator parameters

Architecture:
-------------
FasterKAN consists of a sequence of layers, each containing:
1. A Radial Spline Function (RSF) using tanh-based RBF
2. A dropout layer with rate scaled based on grid count
3. A linear layer (without bias)

The RSF transformation computes:
    f(x) = sech²(σ·(x-μᵢ)) = 1 - tanh²(σ·(x-μᵢ))
where:
    - μᵢ are the grid points
    - σ is the inverse denominator (controlling basis function width)
    - sech² is the squared hyperbolic secant

Example Usage:
-------------
    model = FasterKAN(
        layers_hidden=[784, 100, 10],  # Input, hidden, output dimensions
        num_grids=10,                 # Grid points for RBFs
        grid_min=-3.0,                # Minimum grid value
        grid_max=3.0,                 # Maximum grid value
        inv_denominator=1.0           # Inverse denominator (σ)
    )
    
    # Forward pass
    outputs = model(inputs)

Components:
-------------
- RSWAFFunction: Autograd function for RBF computation
- RSF: Radial Spline Function module used as a wrapper for the RSWAFFunction
- FasterKANLayer: Single Layer combining RSF, dropout, and linear transformation
- FasterKAN: Main model class, that can consists of many different FasterKANLayers
"""

import torch
import torch.nn as nn
from typing import *
from torch.autograd import Function

USE_BIAS_ON_LINEAR = False  # NOTE: Bias must be false to be able to implement on fpga

class RSWAFFunction(Function):
    """
    Autograd function for Radial Spline Wavelet Activation Function.
    
    Computes the derivative of tanh((x-grid)*inv_denominator) with respect to x,
    which is sech²((x-grid)*inv_denominator) = 1 - tanh²((x-grid)*inv_denominator).
    
    The backward pass:
    1. Scales gradients for grid and inv_denominator parameters by 10
    2. Allows selective training of grid and inv_denominator parameters
    """
    @staticmethod
    def forward(ctx, input, grid, inv_denominator):
        """
        Args:
            input (torch.Tensor): Input tensor [batch_size, input_dim]
            grid (torch.Tensor): Grid points [num_grids]
            inv_denominator (torch.Tensor): Inverse denominator
            
        Returns:
            torch.Tensor: sech²((x-grid)*inv_denominator) values
        """
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator) 
        tanh_diff = torch.tanh(diff_mul)
        tanh_diff_deriviative = 1. - tanh_diff ** 2  # sech^2(x) = 1 - tanh^2(x)

        ctx.save_for_backward(inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative) # Save tensors for backward pass

        return tanh_diff_deriviative
    
    @staticmethod
    def backward(ctx, grad_output,train_grid: bool = True, train_inv_denominator: bool = True):
        """
        Args:
            ctx: Context from forward pass
            grad_output (torch.Tensor): Gradient from downstream layers
            train_grid (bool): Whether to compute gradients for grid points
            train_inv_denominator (bool): Whether to compute gradients for inv_denominator
            
        Returns:
            tuple: Gradients for input, grid, and inv_denominator
        """
        inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative = ctx.saved_tensors
        grad_grid = grad_inv_denominator = None
        
        deriv = -2 * inv_denominator * tanh_diff * tanh_diff_deriviative * grad_output

        # Compute the backward pass for the input
        grad_input =  deriv.sum(dim=-1)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator

        # Compute the backward pass for grid
        if ctx.train_grid:
            grad_grid = -10*deriv.sum(dim=-2) # NOTE: We boost the gradient by 10 to make it more significant

        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            grad_inv_denominator = 10*(diff_mul * deriv).sum(0) # NOTE: We boost the gradient by 10 to make it more significant

            if inv_denominator.view(-1).size(0) == 1 :
                grad_inv_denominator = grad_inv_denominator.sum()
                
        return grad_input, grad_grid, grad_inv_denominator

class RSF(nn.Module):
    """
    Args:
        train_grid (bool): Whether to update grid points during training
        train_inv_denominator (bool): Whether to update inv_denominator during training
        grid_min (float): Minimum value for grid points
        grid_max (float): Maximum value for grid points
        num_grids (int): Number of grid points to use
        inv_denominator (float): Initial value for inverse denominator parameter
    
    Attributes:
        grid (nn.Parameter): Learnable grid points evenly spaced from grid_min to grid_max
        inv_denominator (nn.Parameter): Learnable inverse denominator controlling RBF width
    """
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float
    ):
        super(RSF,self).__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)

        self.train_grid = train_grid
        self.train_inv_denominator = train_inv_denominator

        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Transformed tensor [batch_size, input_dim, num_grids]
        """
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator) # returns tanh_diff_derivative


class FasterKANLayer(nn.Module):
    """
    A single layer in the FasterKAN architecture.
    
    The layer applies the following sequence:
    1. Transform inputs using Radial Spline Functions (RSF)
    2. Apply dropout with rate based on grid count (1-0.75^num_grids)
    3. Apply linear transformation to the outputs
    
    Args:
        train_grid (bool): Whether to update grid points during training
        train_inv_denominator (bool): Whether to update inv_denominator during training
        input_dim (int): Dimensionality of input features
        output_dim (int): Dimensionality of output features
        grid_min (float): Minimum value for grid points
        grid_max (float): Maximum value for grid points
        num_grids (int): Number of grid points to use
        inv_denominator (float): Initial value for inverse denominator parameter
    
    Attributes:
        rbf (RSF): Radial Spline Function module
        drop (nn.Dropout): Dropout layer with adaptive rate
        linear (nn.Linear): Linear transformation without bias
    """
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float
    ) -> None:
        super(FasterKANLayer,self).__init__()

        self.rbf = RSF(train_grid, train_inv_denominator,grid_min, grid_max, num_grids, inv_denominator)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        self.drop = nn.Dropout(1-0.75**(num_grids)) # NOTE: Dropout rate increases with num_grids

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_dim]
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        return output


class FasterKAN(nn.Module):
    """
    FasterKAN: Radial Basis Function-based Kolmogorov-Arnold Network.
    This model stacks multiple FasterKANLayers to create a deep RBF-KAN architecture.
    
    Args:
        layers_hidden (List[int]): List of layer dimensions including input and output dimensions
            e.g., [784, 100, 10] for MNIST classification with one hidden layer
        num_grids (Union[int, List[int]]): Number of grid points for each layer
            If a single int is provided, it's used for all layers
        grid_min (float): Minimum value for grid points
        grid_max (float): Maximum value for grid points
        inv_denominator (float): Initial value for inverse denominator parameter
    
    Attributes:
        train_grid (bool): Whether grid points are being updated during training
        train_inv_denominator (bool): Whether inv_denominator is being updated during training
        layers (nn.ModuleList): List of FasterKANLayer modules
        is_eval (bool): Whether the model is in evaluation mode
    
    Example:
        ```python
        model = FasterKAN([784, 100, 10], num_grids=10, grid_min=-3.0, grid_max=3.0, inv_denominator=1.0)
        output = model(input_tensor)  # Shape: [batch_size, 10]
        ```
    """
    def __init__(
        self, layers_hidden: List[int], 
        num_grids: Union[int, List[int]],
        grid_min: float,
        grid_max: float,
        inv_denominator: float
    ):
        super(FasterKAN, self).__init__()

        self.train_grid = True
        self.train_inv_denominator = True

        if not hasattr(num_grids, '__iter__'):
            num_grids = [num_grids for _ in layers_hidden[:-1]]

        if len(num_grids) < len(layers_hidden)-1:
            num_grids = num_grids + [num_grids[-1] for _ in range(len(layers_hidden)-1-len(num_grids))]

        assert len(num_grids) == len(layers_hidden) - 1

        self.layers = nn.ModuleList([
            FasterKANLayer(
                train_grid=self.train_grid,
                train_inv_denominator=self.train_inv_denominator,
                input_dim=in_dim, 
                output_dim=out_dim, 
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids_i,
                inv_denominator=inv_denominator
            ) for _iter, (num_grids_i, in_dim, out_dim) in enumerate(zip(num_grids, layers_hidden[:-1], layers_hidden[1:]))
        ])

    def eval(self):
        """
        Set the model to evaluation mode, disabling grid and inv_denominator parameter updates.
        """
        self.is_eval = True
        self.train_grid = False
        self.train_inv_denominator = False
        super().eval()

    def train(self, mode=True):
        """
        Set the model to training mode, enabling updates to grid and inv_denominator parameters.
        
        Args:
            mode (bool): Whether to enable training mode (True) or evaluation mode (False)
        """
        self.is_eval = not mode
        self.train_grid = mode
        self.train_inv_denominator = mode
        super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x