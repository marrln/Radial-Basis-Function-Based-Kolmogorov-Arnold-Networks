
import torch
import torch.nn as nn
from typing import *
from torch.autograd import Function
from torchinfo import summary 
from thop import profile
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Change the Defaults easily by altering the following:
grid_min_global = -1.2
grid_max_global = 0.2
inv_denominator_global = 0.5 
num_grids_global = 8

class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator):

        # Move tensors to the device of input
        grid = grid.to(input.device)
        inv_denominator = inv_denominator.to(input.device)

        # Compute the forward pass
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator) 
        tanh_diff = torch.tanh(diff)
        tanh_diff_deriviative = -tanh_diff.mul(tanh_diff) + 1  # sech^2(x) = 1 - tanh^2(x)
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator)
        return tanh_diff_deriviative
    
    @staticmethod
    def backward(ctx, grad_output,train_grid: bool = True, train_inv_denominator: bool = True):

        # Retrieve saved tensors
        input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = ctx.saved_tensors
        grad_grid = None
        grad_inv_denominator = None

        # Compute the backward pass for the input
        grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output
        grad_input = grad_input.sum(dim=-1).mul(inv_denominator) 

        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator

        # Compute the backward pass for grid
        if ctx.train_grid:
            grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(dim=0)
            # TODO: Check the alternative method from Github:
            #grad_grid = -(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) 
                    
        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            grad_inv_denominator = (grad_output* diff).sum()
        return grad_input, grad_grid, grad_inv_denominator, None, None # same number as tensors or parameters

class RSF(nn.Module):
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        grid_min: float = grid_min_global,
        grid_max: float = grid_max_global,
        num_grids: int = num_grids_global,
        inv_denominator: float = inv_denominator_global
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool, device=device)
        self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool, device=device) 
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32, device=device), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        x = x.to(self.grid.device)
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator) # returns tanh_diff_derivative

class FasterKANLayer(nn.Module):
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        input_dim: int,
        output_dim: int,
        grid_min: float = grid_min_global,
        grid_max: float = grid_max_global,
        num_grids: int = num_grids_global,
        inv_denominator: float = inv_denominator_global 
    ) -> None:
        super().__init__()
        self.rbf = RSF(train_grid, train_inv_denominator,grid_min, grid_max, num_grids, inv_denominator)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=False) # NOTE: Bias must be false to be able to implement on fpga

    def forward(self, x):
        x = x.view(x.size(0), -1)                       # Shape: [batch_size, input_dim]
        spline_basis = self.rbf(x).view(x.shape[0], -1) # Shape: [batch_size, input_dim * num_grids]
        output = self.linear(spline_basis)              # Shape: [batch_size, output_dim]
        return output

class FasterKAN(nn.Module):
    def __init__(
        self, layers_hidden, 
        num_grids: int = num_grids_global, 
        grid_min: float = grid_min_global, 
        grid_max: float = grid_max_global, 
        inv_denominator: float = inv_denominator_global
    ):
        super(FasterKAN, self).__init__()

        self.train_grid = False
        self.train_inv_denominator = False

        self.layers = nn.ModuleList([
            FasterKANLayer(
                train_grid=self.train_grid,
                train_inv_denominator=self.train_inv_denominator,
                input_dim=in_dim, 
                output_dim=out_dim, 
                grid_min=grid_min, # NOTE: added later
                grid_max=grid_max, # NOTE: added later
                num_grids=num_grids,
                inv_denominator=inv_denominator # NOTE: added later
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
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
        x.to(device)
        for layer in self.layers:
            x = layer(x)
        return x
