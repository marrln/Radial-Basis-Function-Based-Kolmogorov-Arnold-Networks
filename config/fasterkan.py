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

        # # Move tensors to the device of input
        # grid = grid.to(input.device)
        # inv_denominator = inv_denominator.to(input.device)

        # Compute the forward pass
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator) 
        tanh_diff = torch.tanh(diff_mul)
        tanh_diff_deriviative = 1. - tanh_diff ** 2  # sech^2(x) = 1 - tanh^2(x)
        # tanh_diff_deriviative = -tanh_diff.mul(tanh_diff) + 1  # sech^2(x) = 1 - tanh^2(x)
        
        # Save tensors for backward pass
        # ctx.save_for_backward(input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator)
        ctx.save_for_backward(inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative)

        return tanh_diff_deriviative
    
    @staticmethod
    def backward(ctx, grad_output,train_grid: bool = True, train_inv_denominator: bool = True):

        # Retrieve saved tensors
        # input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = ctx.saved_tensors
        inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative = ctx.saved_tensors
        grad_grid = grad_inv_denominator = None
        
        deriv = -2 * inv_denominator * tanh_diff * tanh_diff_deriviative * grad_output

        # Compute the backward pass for the input
        grad_input =  deriv.sum(dim=-1)
        # grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output
        # grad_input = grad_input.sum(dim=-1).mul(inv_denominator) 

        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator

        # Compute the backward pass for grid
        if ctx.train_grid:
            grad_grid = -10*deriv.sum(dim=-2)
            # grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(dim=0)
            # TODO: Check the alternative method from Github:
            #grad_grid = -(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) 
                    
        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            # grad_inv_denominator = (grad_output* diff).sum()
            grad_inv_denominator = 10*(diff_mul * deriv).sum(0)
            
            if inv_denominator.view(-1).size(0) == 1 :
                grad_inv_denominator = grad_inv_denominator.sum()
                
        return grad_input, grad_grid, grad_inv_denominator #, None, None # same number as tensors or parameters

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
        super(RSF,self).__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)

        # self.train_grid = torch.tensor(train_grid, dtype=torch.bool, device=device)
        # self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool, device=device) 

        self.train_grid = train_grid
        self.train_inv_denominator = train_inv_denominator

        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32, device=device), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        # x = x.to(self.grid.device)
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator) # returns tanh_diff_derivative

class RSF2(nn.Module):
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        grid_min: float = grid_min_global,
        grid_max: float = grid_max_global,
        num_grids: int = num_grids_global,
        inv_denominator: float = inv_denominator_global
    ):
        super(RSF2, self).__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)

        # self.train_grid = torch.tensor(train_grid, dtype=torch.bool, device=device)
        # self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool, device=device) 

        self.train_grid = train_grid
        self.train_inv_denominator = train_inv_denominator

        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32, device=device), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Compute the forward pass
        diff_mul = (x[..., None] - self.grid) * self.inv_denominator
        tanh_diff = self.tanh(diff_mul)
        tanh_diff_deriviative = 1. - tanh_diff ** 2  # sech^2(x) = 1 - tanh^2(x)
        
        return tanh_diff_deriviative

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
        super(FasterKANLayer,self).__init__()
        self.rbf = RSF(train_grid, train_inv_denominator,grid_min, grid_max, num_grids, inv_denominator)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=False) # NOTE: Bias must be false to be able to implement on fpga
        self.drop = nn.Dropout(1-0.75**(num_grids))

    # def forward(self, x):
    #     x = x.view(x.size(0), -1)                       # Shape: [batch_size, input_dim]
    #     spline_basis = self.rbf(x).view(x.shape[0], -1) # Shape: [batch_size, input_dim * num_grids]
    #     output = self.linear(spline_basis)              # Shape: [batch_size, output_dim]
    #     return output
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # assert x.shape[1] == self.linear.in_features // self.rbf.grid.shape[0], \
        #     f"Expected input with {self.linear.in_features // self.rbf.grid.shape[0]} features, got {x.shape[1]}"
        spline_basis = self.rbf(x).view(batch_size, -1)
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        # output = self.drop(output)
        
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

        self.train_grid = True
        self.train_inv_denominator = True
        
        if not hasattr(num_grids, '__iter__'):
            num_grids = [num_grids for _ in layers_hidden[:-1]]
            
        if len(num_grids) < len(layers_hidden)-1:
            num_grids = num_grids + [num_grids[-1] for _ in range(len(layers_hidden)-1-len(num_grids))]
            
        # print(num_grids,layers_hidden)
        assert len(num_grids) == len(layers_hidden) -1

        self.layers = nn.ModuleList([
            FasterKANLayer(
                train_grid=self.train_grid,
                train_inv_denominator=self.train_inv_denominator,
                input_dim=in_dim, 
                output_dim=out_dim, 
                grid_min=grid_min, # NOTE: added later
                grid_max=grid_max, # NOTE: added later
                num_grids=num_grids_i,
                inv_denominator=inv_denominator # NOTE: added later
            ) for _iter, (num_grids_i, in_dim, out_dim) in enumerate(zip(num_grids, layers_hidden[:-1], layers_hidden[1:]))
        ])
        # self.tanh = nn.Tanhshrink()

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
        # for layer in self.layers:
        #     out = layer(x)
        #     if x.shape == out.shape:
        #         x = x + out
        #     else :
        #         x = out
        # return x
        # for layer in self.layers[:-1]:
            # x = self.tanh(layer(x))
        # return self.layers[-1](x)
