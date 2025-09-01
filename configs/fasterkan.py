import torch
import torch.nn as nn
from typing import *
from torch.autograd import Function

class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator):

        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator) 
        tanh_diff = torch.tanh(diff_mul)
        tanh_diff_deriviative = 1. - tanh_diff ** 2  # sech^2(x) = 1 - tanh^2(x)

        ctx.save_for_backward(inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative) # Save tensors for backward pass

        return tanh_diff_deriviative
    
    @staticmethod
    def backward(ctx, grad_output,train_grid: bool = True, train_inv_denominator: bool = True):

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
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator) # returns tanh_diff_derivative


class FasterKANLayer(nn.Module):
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
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=False) # NOTE: Bias must be false to be able to implement on fpga
        self.drop = nn.Dropout(1-0.75**(num_grids))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        return output


class FasterKAN(nn.Module):
    def __init__(
        self, layers_hidden, 
        num_grids: int,
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
        
        assert len(num_grids) == len(layers_hidden) -1

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
        self.is_eval = True
        self.train_grid = False
        self.train_inv_denominator = False
        super().eval()

    def train(self, mode=True):
        self.is_eval = not mode
        self.train_grid = mode
        self.train_inv_denominator = mode
        super().train(mode)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x