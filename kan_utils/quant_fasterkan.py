import torch
import torch.nn as nn
from typing import Literal, overload
import copy 

# Local imports
from kan_utils.fasterkan import *

# CONFIGURATION DICTIONARIES
default_dtype = {
    'grid'   : (8, False),
    'scale'  : (8, False),
    'weight' : (8, False),
    'sdff'   : (8, False),
    'actf'   : (7, True),
    'result' : (8, False),
}
default_frac_bits = {
    'grid'   : 4,
    'scale'  : 4,
    'weight' : 4,
    'sdff'   : 4,
    'actf'   : 7,
    'result' : 4,
}


def extract_range(tensor: torch.Tensor, bit_width: int, unsigned: bool = False) -> int:
    """
    Calculates the number of bits available for quantization based on the maximum absolute value in the input tensor.
    Args:
        tensor (torch.Tensor): Input tensor whose range is to be analyzed.
        bit_width (int): Total number of bits available for quantization.
        unsigned (bool, optional): If True, treats the values as unsigned. Defaults to False.
    Returns:
        int: Number of bits available for representing the quantized values.
    """
    
    return bit_width - int(torch.abs(tensor).max().log2().ceil().cpu()) - int(not unsigned)


def get_dtype(bit_width: int, unsigned: bool = False) -> torch.dtype:
    """
    Returns the appropriate PyTorch dtype for a given bit width and signedness.
    Args:
        bit_width (int): The number of bits required for the dtype.
        unsigned (bool, optional): Whether to use an unsigned dtype. Defaults to False.
    Returns:
        torch.dtype: The corresponding PyTorch dtype.
    """
    
    num_bytes = (bit_width+7) // 8
    corr_num_bytes = 1
    while num_bytes > corr_num_bytes:
        corr_num_bytes *= 2
    
    corr_num_bytes = f'int{int(corr_num_bytes*8)}'
    if unsigned:
        corr_num_bytes = 'u'+corr_num_bytes
        
    return getattr(torch,corr_num_bytes)


def get_bit_width(dtype: torch.dtype):
    """
    Returns the bit width of a given PyTorch data type.
    Args:
        dtype (torch.dtype): The PyTorch data type (e.g., torch.int8, torch.int16).
    Returns:
        str: The bit width as a string (e.g., '8', '16').
    """
    
    dtype = str(dtype).split('.')[-1]
    bits = dtype.split('int')[-1]
    return bits

def resolve_dtype(bits_0: int, unsigned_0: bool, bits_1: int, unsigned_1: bool, reduction: Literal['sum', 'mlt'] = 'sum'):
    """
    Determine the resulting bit width and signedness after combining two data types.
    Args:
        bits_0 (int): Bit width of the first operand.
        unsigned_0 (bool): Whether the first operand is unsigned.
        bits_1 (int): Bit width of the second operand.
        unsigned_1 (bool): Whether the second operand is unsigned.
        reduction (Literal['sum', 'mlt'], optional): Operation type, either 'sum' or 'mlt' (multiply). Defaults to 'sum'.
    Returns:
        Tuple[int, bool]: Resulting bit width and unsigned flag.
    """
    
    unsigned = unsigned_0 and unsigned_1
    
    if not unsigned:
        if unsigned_0:
            bits_0 += 1
        if unsigned_1:
            bits_1 += 1
            
    if reduction == 'mlt' :
        bits = bits_0 + bits_1
    else : # default = 'sum'
        bits = max(bits_0, bits_1)
    
    return bits, unsigned

def quantize_fixed_point(tensor: torch.Tensor, frac_bits: int, bit_width: int, unsigned: bool = False) -> torch.Tensor:
    """
    Quantizes a tensor to fixed-point representation.
    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        frac_bits (int): Number of fractional bits.
        bit_width (int): Total bit width for quantization.
        unsigned (bool, optional): If True, use unsigned quantization. Defaults to False.
    Returns:
        torch.Tensor: Quantized tensor in fixed-point format.
    """
    
    scale = 2 ** frac_bits
    if unsigned:
        max_val = 2 ** (bit_width)-1
        min_val = 0
    else:
        max_val = 2 ** (bit_width-1)-1
        min_val = -2 ** (bit_width-1)
    quantized = torch.clamp((tensor * scale).round(), min_val, max_val).to(get_dtype(bit_width, unsigned))
    return quantized

def dequantize_fixed_point(tensor: torch.Tensor, frac_bits: int = 0) -> torch.Tensor:
    """
    Dequantizes a fixed-point tensor to floating point.
    Args:
        tensor (torch.Tensor): The quantized tensor.
        frac_bits (int, optional): Number of fractional bits used in quantization. Defaults to 0.
    Returns:
        torch.Tensor: The dequantized tensor as float32.
    """
    
    return tensor.to(torch.float32) / (2 ** frac_bits)



# -------------------------------------------------------------------------------------------------------------------
''' QUANTIZED FASTERKAN MODEL '''

class FixedPointFasterKANLayer(nn.Module):   
    @overload
    def __init__(self, fasterKANLayer: FasterKANLayer, dtype_dict: dict = default_dtype, frac_bits_dict: dict = default_frac_bits, hardtanh: bool = False, collect_stats: bool = False):
        ...
        
    @overload
    def __init__(self, input_dim: int, output_dim: int, num_grids: int, dtype_dict: dict = default_dtype, frac_bits_dict: dict = default_frac_bits, hardtanh: bool = False, collect_stats: bool = False):
        ...

    def __init__(self, dtype_dict: dict = default_dtype, frac_bits_dict: dict = default_frac_bits, hardtanh: bool = False, collect_stats: bool = False, **kwargs):
        """
        Initializes the FixedPointFasterKANLayer with optional quantized parameters.
        Args:
            dtype_dict (dict, optional): Dictionary specifying data types for quantization. Defaults to default_dtype.
            frac_bits_dict (dict, optional): Dictionary specifying fractional bits for quantization. Defaults to default_frac_bits.
            hardtanh (bool, optional): Whether to use hardtanh activation. Defaults to False.
            collect_stats (bool, optional): Whether to collect statistics during forward pass. Defaults to False.
            **kwargs: Additional keyword arguments. If 'fasterKANLayer' is provided, initializes from its parameters; 
                otherwise, expects 'input_dim', 'output_dim', and 'num_grids' for layer initialization.
        """
        super(FixedPointFasterKANLayer, self).__init__()
    
        self.__common_init(dtype_dict=dtype_dict, frac_bits_dict=frac_bits_dict, hardtanh=hardtanh, collect_stats=collect_stats)
        
        if 'fasterKANLayer' in kwargs.keys():
            fasterKANLayer = kwargs['fasterKANLayer']
            self.grid = torch.nn.Parameter(
                quantize_fixed_point(fasterKANLayer.rbf.grid.data, self.frac_bits_dict['grid'], *self.dtype_dict['grid']),
                requires_grad=False
            )
            self.inv_denom = torch.nn.Parameter(
                quantize_fixed_point(fasterKANLayer.rbf.inv_denominator.data, self.frac_bits_dict['scale'], *self.dtype_dict['scale']),
                requires_grad=False
            )
            self.weight = torch.nn.Parameter(
                quantize_fixed_point(fasterKANLayer.linear.weight.data, self.frac_bits_dict['weight'], *self.dtype_dict['weight']),
                requires_grad=False
            )
        else :
            input_dim, output_dim, num_grids = kwargs['input_dim'],kwargs['output_dim'],kwargs['num_grids']
            self.grid = torch.nn.Parameter(
                torch.empty((num_grids), dtype=get_dtype(*self.dtype_dict['grid'])),
                requires_grad=False
            )
            self.inv_denom = torch.nn.Parameter(
                torch.empty((), dtype=get_dtype(*self.dtype_dict['scale'])),
                requires_grad=False
            )
            self.weight = torch.nn.Parameter(
                torch.empty((output_dim,input_dim * num_grids,), dtype=get_dtype(*self.dtype_dict['weight'])),
                requires_grad=False
            )


    def update_hardtanh(self, hardtanh: bool = False):
        """
        Updates the activation function to either Hardtanh or Tanh based on the 'hardtanh' flag.
        Args:
            hardtanh (bool): If True, sets activation to Hardtanh with quantized limits; otherwise uses Tanh.
        """
        
        self.hardtanh = hardtanh
        if self.hardtanh :
            self.one = 2**self.frac_bits_dict['actf_int']
            self.tanh = nn.Hardtanh(-self.one,self.one)
        else:
            self.one = 1.
            self.tanh = nn.Tanh()


    def update_dtype_dict(self, dtype_dict: dict):
        """
        Update the internal dtype dictionary with new values and recalculate related dtype fields.
        Args:
            dtype_dict (dict): Dictionary containing dtype information to update.
        Updates:
            - Merges new dtype information into self.dtype_dict.
            - Computes and updates derived dtype fields ('sdff_int', 'actf_int', 'matmul').
            - Updates self.torch_dtype with resolved torch dtypes.
        """
        
        self.dtype_dict.update(dtype_dict)
        
        dtype_matmul, unsigned_matmul = resolve_dtype(*self.dtype_dict['actf'],*self.dtype_dict['weight'],reduction='mlt')
        dtype_matmul += 8
        
        self.dtype_dict.update({
            'sdff_int' : resolve_dtype(*self.dtype_dict['grid'],*self.dtype_dict['scale'],reduction='mlt'),
            'actf_int' : resolve_dtype(*self.dtype_dict['sdff'],*self.dtype_dict['sdff'],reduction='mlt'),
            'matmul' : (dtype_matmul,unsigned_matmul),
        })
        self.torch_dtype = {
            key : get_dtype(*value)
                for key, value in self.dtype_dict.items()
        }


    def update_frac_bits(self, frac_bits_dict: dict):
        """
        Update the fractional bits configuration for quantization.
        Args:
            frac_bits_dict (dict): Dictionary containing new fractional bits settings.
        This method updates the internal fractional bits dictionary, applies the changes,
        and updates the hardtanh activation accordingly.
        """
        
        self.frac_bits_dict.update(frac_bits_dict)
        self.fix_frac_bits()
        self.update_hardtanh(self.hardtanh)
            

    def quantize(self, fasterKANLayer: FasterKANLayer):
        """
        Quantizes the parameters of the given FasterKANLayer using fixed-point representation.
        Args:
            fasterKANLayer (FasterKANLayer): The layer whose parameters will be quantized.
        This method updates the grid, inv_denom, and weight attributes with their quantized values.
        """
        
        self.grid.data = quantize_fixed_point(fasterKANLayer.rbf.grid.data.detach().clone(), self.frac_bits_dict['grid'], *self.dtype_dict['grid'])
        self.inv_denom.data = quantize_fixed_point(fasterKANLayer.rbf.inv_denominator.data.detach().clone(), self.frac_bits_dict['scale'], *self.dtype_dict['scale'])
        self.weight.data = quantize_fixed_point(fasterKANLayer.linear.weight.data.detach().clone(), self.frac_bits_dict['weight'], *self.dtype_dict['weight'])


    def state_dict(self, destination: Optional[dict] = None, prefix: str = '', keep_vars: bool = False, *args, **kwargs):
        """
        Returns the state of the module as a dictionary, including custom quantization attributes.
        Args:
            destination (Optional[dict]): Destination dictionary to populate. Defaults to None.
            prefix (str): Prefix for parameter and buffer names. Defaults to ''.
            keep_vars (bool): Whether to keep variables instead of Tensors. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            dict: A dictionary containing the module's state, including dtype_dict, frac_bits_dict, and hardtanh.
        """
        
        original_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars, *args, **kwargs)
        original_dict[prefix + 'dtype_dict'] = self.dtype_dict
        original_dict[prefix + 'frac_bits_dict'] = self.frac_bits_dict
        original_dict[prefix + 'hardtanh'] = self.hardtanh
        
        return original_dict


    def _load_from_state_dict(        
        self, 
        state_dict : dict, 
        prefix: str, 
        local_metadata: dict, 
        strict: bool, 
        missing_keys: list, 
        unexpected_keys: list, 
        error_msgs: list,
    ):
        """
        Loads the module's state from the given state_dict, including custom attributes
        such as dtype_dict, frac_bits_dict, and hardtanh. Updates the module's internal
        state and handles missing or unexpected keys accordingly.
        Args:
            state_dict (dict): State dictionary containing parameters and buffers.
            prefix (str): Prefix for parameter and buffer names.
            local_metadata (dict): Metadata for the local module.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the module's keys.
            missing_keys (list): List to append missing keys to.
            unexpected_keys (list): List to append unexpected keys to.
            error_msgs (list): List to append error messages to.
        """
        
        # Load the additional attribute from the state_dict
        self.dtype_dict = copy.deepcopy(state_dict.get(prefix+'dtype_dict', default_dtype))
        self.frac_bits_dict = copy.deepcopy(state_dict.get(prefix+'frac_bits_dict', default_frac_bits))
        self.hardtanh = state_dict.get(prefix+'hardtanh', False)
        
        self.update_dtype_dict(self.dtype_dict)
        self.update_frac_bits(self.frac_bits_dict)
        self.update_hardtanh(self.hardtanh)
        
        # Load the original state_dict
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs,)
        
        for local_dict in ['dtype_dict','frac_bits_dict','hardtanh']:
            if prefix + local_dict in unexpected_keys:
                unexpected_keys.pop(unexpected_keys.index(prefix + local_dict))
            if prefix + local_dict not in state_dict:
                missing_keys.append(prefix + local_dict)
            
            

    def __common_init(
        self, 
        dtype_dict: dict = default_dtype, 
        frac_bits_dict: dict = default_frac_bits, 
        hardtanh: bool = False, 
        collect_stats: bool = False
        ):
        """
        Initializes common quantization parameters for the module.
        Args:
            dtype_dict (dict, optional): Dictionary specifying data types for quantization. Defaults to default_dtype.
            frac_bits_dict (dict, optional): Dictionary specifying fractional bits for quantization. Defaults to default_frac_bits.
            hardtanh (bool, optional): Whether to use hardtanh activation. Defaults to False.
            collect_stats (bool, optional): Whether to collect statistics (min/max tracking). Defaults to False.
        """
        
        self.dtype_dict = copy.deepcopy(default_dtype)
        self.frac_bits_dict = copy.deepcopy(default_frac_bits)
        self.hardtanh = False
        self.track_max = float('-inf')
        self.track_min = float('+inf')
        
        self.update_dtype_dict(copy.deepcopy(dtype_dict))
        self.update_frac_bits(copy.deepcopy(frac_bits_dict))
        self.update_hardtanh(hardtanh)
        
        self.collect_stats = collect_stats


    def fit_frac_bits(self, x: torch.Tensor, fasterKANLayer: FasterKANLayer):
        """
        Computes and updates the fractional bit allocation for various intermediate tensors 
        in the quantization process of a FasterKANLayer, based on the input tensor `x`.
        Args:
            x (torch.Tensor): Input tensor to the layer.
            fasterKANLayer (FasterKANLayer): The layer whose quantization parameters are being fitted.
        Returns:
            torch.Tensor: The output tensor after passing through the quantization process.
        """
        
        with torch.no_grad():
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
            diff = (x[..., None] - fasterKANLayer.rbf.grid)
            
            self.frac_bits_dict['grid']  = max(
                extract_range(x, *self.dtype_dict['grid']),
                extract_range(fasterKANLayer.rbf.grid, *self.dtype_dict['grid']),
                extract_range(diff, *self.dtype_dict['grid']),
            )
            self.frac_bits_dict['scale'] = extract_range(fasterKANLayer.rbf.inv_denominator, *self.dtype_dict['scale'])
            
            sdff = diff * fasterKANLayer.rbf.inv_denominator
            self.frac_bits_dict['sdff'] = extract_range(sdff, *self.dtype_dict['sdff'])

            act = 1. - torch.tanh(sdff) ** 2
            self.frac_bits_dict['actf'] = extract_range(act, *self.dtype_dict['actf'])
            self.frac_bits_dict['weight'] = extract_range(fasterKANLayer.linear.weight, *self.dtype_dict['weight'])

            output = fasterKANLayer.linear(act.view(batch_size,-1))
            self.frac_bits_dict['result'] = extract_range(output, *self.dtype_dict['result'])

            self.fix_frac_bits()
        
        return output
        
        
    def fix_frac_bits(self):
        """
        Updates the fractional bits dictionary with derived values for internal quantization parameters.
        This method computes and sets the fractional bits for 'sdff_int', 'actf_int', and 'matmul'
        based on existing values in `self.frac_bits_dict`.
        """
        
        self.frac_bits_dict.update({
            'sdff_int' : self.frac_bits_dict['grid'] + self.frac_bits_dict['scale'],
            'actf_int' : self.frac_bits_dict['sdff'] * 2,
            'matmul' : self.frac_bits_dict['actf'] + self.frac_bits_dict['weight'],
        })
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the quantized RBF-KAN layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
        Returns:
            torch.Tensor: Output tensor after quantized activation and linear transformation.
        """
        
        batch_size = x.size(0)
        
        
        # == RSWAF with proper scaling ==
        # 1. Operation: Diff = input - grid
        x_int   = x.view(batch_size, -1).to(self.torch_dtype['sdff_int']).unsqueeze(-1)
        grid_int = self.grid.to(self.torch_dtype['sdff_int'])
        diff    = x_int - grid_int
        scale = self.inv_denom.to(self.torch_dtype['sdff_int'])        
        
        # 2. Operation: Scaled = diff * inv_denom
        sdff = torch.bitwise_right_shift(
            diff * scale, 
            self.frac_bits_dict['sdff_int'] - self.frac_bits_dict['sdff']
        ).to(self.torch_dtype['sdff'])
        
        # 3. Operation: 1-tanh^2(x)
        if self.hardtanh:
            sdff = sdff.to(self.torch_dtype['actf_int'])
        else :
            sdff = sdff.float() / (2 ** self.frac_bits_dict['sdff']) 
            
        act = self.one - self.tanh(sdff) ** 2
        
        if self.hardtanh:
            act =torch.bitwise_right_shift(act, 
                self.frac_bits_dict['actf_int'] - self.frac_bits_dict['actf']).to(self.torch_dtype['actf'])
        else :
            act = quantize_fixed_point(act, self.frac_bits_dict['actf'], *self.dtype_dict['actf'])
        
        
        # == Linear layer with careful output scaling ==
        act_reshaped = act.view(batch_size, -1).to(self.torch_dtype['matmul'])
        weights = self.weight.to(self.torch_dtype['matmul'])     
        
        output = torch.nn.functional.linear(act_reshaped, weights)
        output = torch.bitwise_right_shift(output, 
                    self.frac_bits_dict['matmul'] - self.frac_bits_dict['result']).to(self.torch_dtype['result'])
        
        if (self.collect_stats):
            self.track_max = max(self.track_max, output.max())
            self.track_min = min(self.track_min, output.min())
        

        # # DEBUG PRINTS:
        # print(f"Input range: {x_q.min().item()} to {x_q.max().item()}")
        # print(f"Scaled range: {scaled.min().item()} to {scaled.max().item()}")
        # print(f"Diff range: {diff.min().item()} to {diff.max().item()}")
        # print(f"Scaled_fp range: {scaled_fp.min().item()} to {scaled_fp.max().item()}")
        # print(f"Activation range: {act.min().item()} to {act.max().item()}")
        # print(f"act_q range: {act_q.min().item()} to {act_q.max().item()}")
        # print(f"Pre-shift output range: {output.min().item()} to {output.max().item()}")
        
        return output



class FixedPointFasterKAN(nn.Module):
    """
    A quantized version of FasterKAN using fixed-point arithmetic.
    This module wraps a FasterKAN model or constructs a new one, enabling quantization
    with configurable data types and fractional bits per layer. It supports quantization,
    fractional bit fitting, and hardtanh activation control for efficient inference.
    Args:
        model (FasterKAN, optional): Pretrained FasterKAN model to quantize.
        layers_hidden (list, optional): List of hidden layer sizes for new model construction.
        num_grids (int or list, optional): Number of grids per layer.
        dtype_dict (dict or list, optional): Data type(s) for quantization per layer.
        frac_bits_dict (dict or list, optional): Fractional bits for quantization per layer.
        hardtanh (bool, optional): Whether to use hardtanh activation.
        collect_stats (bool, optional): Whether to collect statistics for quantization.
    Methods:
        quantize(fasterKAN): Quantizes the model parameters from a FasterKAN instance.
        fit_frac_bits(x, fasterKAN): Fits optimal fractional bits per layer using input data.
        fit_quantize(x, fasterKAN): Fits and applies quantization to match layer interfaces.
        forward(x): Forward pass through the quantized network.
        update_hardtanh(hardtanh): Updates the hardtanh activation setting for all layers.
    """    
    
    
    @overload
    def __init__(
        self, 
        model: FasterKAN,
        dtype_dict: dict = default_dtype,
        frac_bits_dict: dict = default_frac_bits,
        hardtanh: bool = False,
        collect_stats: bool = False
    ):
        ...
        
    @overload
    def __init__(
        self, 
        layers_hidden: list, 
        num_grids: int = 1,
        dtype_dict: dict = default_dtype,
        frac_bits_dict: dict = default_frac_bits,
        hardtanh: bool = False,
        collect_stats: bool = False
    ):
        ...
    
    def __init__(
        self, 
        model: FasterKAN = None,
        layers_hidden: list = [],
        num_grids: int = 1,
        dtype_dict: dict = default_dtype,
        frac_bits_dict: dict = default_frac_bits,
        hardtanh: bool = False,
        collect_stats: bool = False,
        **kwargs
    ):
        """
        Initializes the FixedPointFasterKAN module.
        Args:
            model (FasterKAN, optional): Predefined FasterKAN model to quantize. If None, a new model is constructed.
            layers_hidden (list, optional): List specifying the sizes of hidden layers for new model construction.
            num_grids (int or list, optional): Number of grids per layer or list of grids for each layer.
            dtype_dict (dict or list, optional): Data type configuration(s) for each layer.
            frac_bits_dict (dict or list, optional): Fractional bits configuration(s) for each layer.
            hardtanh (bool, optional): Whether to use hardtanh activation.
            collect_stats (bool, optional): Whether to collect quantization statistics.
            **kwargs: Additional keyword arguments.
        Raises:
            AssertionError: If the lengths of dtype_dict or frac_bits_dict do not match the number of layers.
        """
        
        super(FixedPointFasterKAN, self).__init__()
        
        if model is not None:
            
            if not isinstance(dtype_dict, (list, tuple)):
                dtype_dict = [dtype_dict for _ in model.layers]
                
            if len(dtype_dict) < len(model.layers)-1:
                dtype_dict = dtype_dict + [dtype_dict[-1] for _ in range(len(model.layers)-len(dtype_dict))]
                
            if not isinstance(frac_bits_dict, (list, tuple)) :
                frac_bits_dict = [frac_bits_dict for _ in model.layers]
                
            if len(frac_bits_dict) < len(model.layers):
                frac_bits_dict = frac_bits_dict + [frac_bits_dict[-1] for _ in range(len(model.layers)-len(frac_bits_dict))]
                
            assert len(dtype_dict) == len(model.layers) 
            assert len(frac_bits_dict) == len(model.layers) 

            self.layers = nn.ModuleList([
                FixedPointFasterKANLayer(
                    fasterKANLayer=layer,
                    dtype_dict=dtype_dict_i,
                    frac_bits_dict=frac_bits_dict_i,
                    hardtanh=hardtanh,
                    collect_stats=collect_stats,
                ) for layer, dtype_dict_i, frac_bits_dict_i in zip(
                    model.layers, 
                    dtype_dict,
                    frac_bits_dict,
                )
            ])
        else :
            if not hasattr(num_grids, '__iter__'):
                num_grids = [num_grids for _ in layers_hidden[:-1]]
                
            if len(num_grids) < len(layers_hidden)-1:
                num_grids = num_grids + [num_grids[-1] for _ in range(len(layers_hidden)-1-len(num_grids))]
                
            if not isinstance(dtype_dict, (list, tuple)):
                dtype_dict = [dtype_dict for _ in layers_hidden[:-1]]
                
            if len(dtype_dict) < len(layers_hidden)-1:
                dtype_dict = dtype_dict + [dtype_dict[-1] for _ in range(len(layers_hidden)-1-len(dtype_dict))]
                
            if not isinstance(frac_bits_dict, (list, tuple)) :
                frac_bits_dict = [frac_bits_dict for _ in layers_hidden[:-1]]
                
            if len(frac_bits_dict) < len(layers_hidden)-1:
                frac_bits_dict = frac_bits_dict + [frac_bits_dict[-1] for _ in range(len(layers_hidden)-1-len(frac_bits_dict))]
                
            assert len(num_grids) == len(layers_hidden) -1
            assert len(dtype_dict) == len(layers_hidden) -1
            assert len(frac_bits_dict) == len(layers_hidden) -1

            self.layers = nn.ModuleList([
                FixedPointFasterKANLayer(
                    input_dim=in_dim, 
                    output_dim=out_dim, 
                    num_grids=num_grids_i,
                    dtype_dict=dtype_dict_i,
                    frac_bits_dict=frac_bits_dict_i,
                    hardtanh=hardtanh,
                    collect_stats=collect_stats,
                ) for in_dim, out_dim, num_grids_i, dtype_dict_i, frac_bits_dict_i in zip(
                    layers_hidden[:-1], 
                    layers_hidden[1:],
                    num_grids, 
                    dtype_dict,
                    frac_bits_dict,
                )
            ])


    def quantize(self, fasterKAN: FasterKAN):
        """
        Quantizes the layers of the current model using the corresponding layers from a given FasterKAN model.
        Args:
            fasterKAN (FasterKAN): The source model whose layers are used for quantization.
        Returns:
            self: The quantized model instance.
        """
        
        for qlayer, ulayer in zip(self.layers, fasterKAN.layers):
            qlayer.quantize(ulayer)
        
        return self

    def fit_frac_bits(self, x: torch.Tensor, fasterKAN: FasterKANLayer):
        """
        Adjusts the fractional bits for each quantization layer to match the corresponding FasterKAN layer.
        Args:
            x: Input tensor to be quantized.
            fasterKAN (FasterKANLayer): Reference FasterKAN layer for determining optimal fractional bits.
        Returns:
            self: The quantization layer instance with updated fractional bits.
        """
        
        output = x
        for qlayer, fasterKANLayer in zip(self.layers, fasterKAN.layers):
            output = qlayer.fit_frac_bits(output,fasterKANLayer)
        
        return self
    
    def fit_quantize(self, x: torch.Tensor, fasterKAN: FasterKAN):
        """
        Quantizes the FasterKAN model by determining optimal fractional bits for each layer based on input data.
        Performs a two-pass process: first, it fits the fractional bits per layer using the input tensor `x`, 
        then aligns the fractional bits between sequential layers to ensure consistency. 
        Finally, applies quantization to the model.
        Args:
            x (torch.Tensor): Input tensor for fitting quantization parameters.
            fasterKAN (FasterKAN): The model to be quantized.
        Returns:
            Quantized model or quantized representation as defined by self.quantize().
        """
        
        # 1st pass to get best per layer
        self.fit_frac_bits(x, fasterKAN)
        
        # Match input & output of sequential layers
        if len(self.layers) > 1:
            for layer_0, layer_1 in zip(self.layers[:-1], self.layers[1:]):
                frac_bits_0 = layer_0.frac_bits_dict['result']
                frac_bits_1 = layer_1.frac_bits_dict['grid']
                frac_bits = min(frac_bits_0, frac_bits_1)
                
                layer_0.frac_bits_dict['result'] = layer_1.frac_bits_dict['grid'] = frac_bits
    
            for layer in self.layers:
                layer.fix_frac_bits()
        
        return self.quantize(fasterKAN)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        for layer in self.layers:
            x = layer(x)
        return x


    def update_hardtanh(self, hardtanh: bool = False):
        """
        Updates the Hardtanh activation setting for all layers in the model.
        Args:
            hardtanh (bool): If True, enables Hardtanh activation; otherwise disables it.
        """        
        for layer in self.layers:
            layer.update_hardtanh(hardtanh)
    
    

class FloatWrapperModule(nn.Module):
    def __init__(self, model: FixedPointFasterKAN):
        """
        Wraps a FixedPointFasterKAN so it can be evaluated like a normal float32 model. 
        It quantizes the inputs, runs the quantized (integer) model, and dequantizes 
        the outputs back to float32, making it compatible with standard loss functions 
        and validation pipelines.
        
        Args:
            model (FixedPointFasterKAN): The model to be wrapped as a float32 model for validation pipeline.
        """     
        
        super(FloatWrapperModule, self).__init__()
        self.model = model
        
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
        
    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
        
    def forward(self,x):
        x = quantize_fixed_point(x, self.model.layers[0].frac_bits_dict['grid'], *self.model.layers[0].dtype_dict['grid'])
        return dequantize_fixed_point(self.model(x), self.model.layers[-1].frac_bits_dict['result'])
        
    def update_hardtanh(self, hardtanh: bool = False):
        """
        Updates the Hardtanh activation setting for all layers in the model.
        Args:
            hardtanh (bool): If True, enables Hardtanh activation; otherwise disables it.
        """        
        for layer in self.layers:
            layer.update_hardtanh(hardtanh)
        