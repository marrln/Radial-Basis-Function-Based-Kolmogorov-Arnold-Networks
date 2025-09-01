import torch
import torch.nn as nn
from typing import Literal, overload

from fasterkan import *

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

def extract_range(x, bit_width, unsigned=False):
    return bit_width - int(torch.abs(x).max().log2().ceil().cpu()) - int(not unsigned)

def get_dtype(bit_width, unsigned=False):
    num_bytes = (bit_width+7) // 8
    corr_num_bytes = 1
    while num_bytes > corr_num_bytes:
        corr_num_bytes *= 2
    
    corr_num_bytes = f'int{int(corr_num_bytes*8)}'
    if unsigned:
        corr_num_bytes = 'u'+corr_num_bytes
        
    # print(f'DEBUG get_dtypes({bit_width}, {unsigned}) = {getattr(torch,corr_num_bytes)} -> {corr_num_bytes}')
        
    return getattr(torch,corr_num_bytes)

def get_bit_width(dtype):
    dtype = str(dtype).split('.')[-1]
    bits = dtype.split('int')[-1]
    return bits

def resolve_dtype(bits_0, unsigned_0, bits_1, unsigned_1, reduction : Literal['sum','mlt'] = 'sum'):
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

def quantize_fixed_point(tensor, frac_bits, bit_width, unsigned=False):
    scale = 2 ** frac_bits
    if unsigned:
        max_val = 2 ** (bit_width)-1
        min_val = 0
    else:
        max_val = 2 ** (bit_width-1)-1
        min_val = -2 ** (bit_width-1)
    quantized = torch.clamp((tensor * scale).round(), min_val, max_val).to(get_dtype(bit_width, unsigned))
    return quantized

def dequantize_fixed_point(tensor, frac_bits=0):
    return tensor.to(torch.float32) / (2 ** frac_bits)

# -------------------------------------------------------------------------------------------------------------------
''' QUANTIZED FASTERKAN MODEL '''

class FixedPointFasterKANLayer(nn.Module):
    @overload
    def __init__(self, fasterKANLayer: FasterKANLayer, dtype_dict=default_dtype, frac_bits_dict=default_frac_bits, hardtanh=False, collect_stats = False):
        ...
        
    @overload
    def __init__(self, input_dim, output_dim, num_grids, dtype_dict=default_dtype, frac_bits_dict=default_frac_bits, hardtanh=False, collect_stats = False):
        ...
    
    def __init__(self, dtype_dict=default_dtype, frac_bits_dict=default_frac_bits, hardtanh=False, collect_stats = False, **kwargs):
        super(FixedPointFasterKANLayer,self).__init__()
        self.__common_init(dtype_dict=dtype_dict, frac_bits_dict=frac_bits_dict, hardtanh=hardtanh, collect_stats = collect_stats)
        
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
        
    def update_hardtanh(self,hardtanh=False):
        self.hardtanh = hardtanh
        if self.hardtanh :
            self.one = 2**self.frac_bits_dict['actf_int']
            self.tanh = nn.Hardtanh(-self.one,self.one)
        else:
            self.one = 1.
            self.tanh = nn.Tanh()
        
    def update_dtype_dict(self, dtype_dict) : 
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
        
    def update_frac_bits(self, frac_bits_dict) : 
        self.frac_bits_dict.update(frac_bits_dict)
        self.fix_frac_bits()
        self.update_hardtanh(self.hardtanh)
            
    def quantize(self, fasterKANLayer: FasterKANLayer):
        self.grid.data = quantize_fixed_point(fasterKANLayer.rbf.grid.data.detach().clone(), self.frac_bits_dict['grid'], *self.dtype_dict['grid'])
        self.inv_denom.data = quantize_fixed_point(fasterKANLayer.rbf.inv_denominator.data.detach().clone(), self.frac_bits_dict['scale'], *self.dtype_dict['scale'])
        self.weight.data = quantize_fixed_point(fasterKANLayer.linear.weight.data.detach().clone(), self.frac_bits_dict['weight'], *self.dtype_dict['weight'])
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, *args, **kwargs):
        original_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars, *args, **kwargs)

        original_dict[prefix + 'dtype_dict'] = self.dtype_dict
        original_dict[prefix + 'frac_bits_dict'] = self.frac_bits_dict
        original_dict[prefix + 'hardtanh'] = self.hardtanh
        
        return original_dict
    
    def _load_from_state_dict(
        self,
        state_dict : dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Load the additional attribute from the state_dict
        self.dtype_dict = state_dict.get(prefix+'dtype_dict', default_dtype)
        self.frac_bits_dict = state_dict.get(prefix+'frac_bits_dict', default_frac_bits)
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
            
    def __common_init(self, dtype_dict=default_dtype, frac_bits_dict=default_frac_bits, hardtanh=False, collect_stats = False):
        self.dtype_dict = default_dtype
        self.frac_bits_dict = default_frac_bits
        self.hardtanh = False
        self.track_max = float('-inf')
        self.track_min = float('+inf')
        
        self.update_dtype_dict(dtype_dict)
        self.update_frac_bits(frac_bits_dict)
        self.update_hardtanh(hardtanh)
        
        self.collect_stats = collect_stats
      
    def fit_frac_bits(self, x, fasterKANLayer: FasterKANLayer):
        with torch.no_grad():
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
            output = (x[..., None] - fasterKANLayer.rbf.grid)
            # self.frac_bits_dict['grid']  = extract_range(output, *self.dtype_dict['grid'])
            self.frac_bits_dict['grid']  = max(
                extract_range(x, *self.dtype_dict['grid']),
                extract_range(fasterKANLayer.rbf.grid, *self.dtype_dict['grid']),
                extract_range(output, *self.dtype_dict['grid']),
            )
            self.frac_bits_dict['scale'] = extract_range(fasterKANLayer.rbf.inv_denominator, *self.dtype_dict['scale'])
            # print('diff :',output.abs().min(),output.abs().max())
            
            output = output * fasterKANLayer.rbf.inv_denominator
            # self.frac_bits_dict['sdff'] = self.frac_bits_dict['grid']
            self.frac_bits_dict['sdff'] = extract_range(output, *self.dtype_dict['sdff'])
            # print('sdff :',output.abs().min(),output.abs().max())
            
            output = 1. - torch.tanh(output) ** 2
            self.frac_bits_dict['actf'] = extract_range(output, *self.dtype_dict['actf'])
            # self.frac_bits_dict['weight'] = self.frac_bits_dict['grid']
            self.frac_bits_dict['weight'] = extract_range(fasterKANLayer.linear.weight, *self.dtype_dict['weight'])
            # self.frac_bits_dict['weight'] = min(extract_range(fasterKANLayer.linear.weight, *self.dtype_dict['weight']), self.dtype_dict['weight'][0])
            # print('actf :',output.abs().min(),output.abs().max())
            
            output = fasterKANLayer.linear(output.view(batch_size,-1))
            self.frac_bits_dict['result'] = extract_range(output, *self.dtype_dict['result'])
            # print('output :',output.abs().min(),output.abs().max())

            self.fix_frac_bits()
        
        return output
        
    def fix_frac_bits(self):
        self.frac_bits_dict.update({
            'sdff_int' : self.frac_bits_dict['grid'] + self.frac_bits_dict['scale'],
            'actf_int' : self.frac_bits_dict['sdff'] * 2,
            'matmul' : self.frac_bits_dict['actf'] + self.frac_bits_dict['weight'],
        })
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # == RSWAF with proper scaling
        # 1. Operation: Diff = input - grid
        x = x.unsqueeze(-1)
        diff = (x - self.grid).to(self.torch_dtype['sdff_int'])
        scale = self.inv_denom.to(self.torch_dtype['sdff_int'])
        # print(diff)
                
        # 2. Operation: Scaled = diff * inv_denom, with int32 to avoid overflow
        sdff = torch.bitwise_right_shift(
            diff * scale, 
            self.frac_bits_dict['sdff_int'] - self.frac_bits_dict['sdff']
        ).to(self.torch_dtype['sdff'])
        # print(sdff)
                
        # 3. Operation: 1-tanh^2(x), with float32 values
        # print('hardtanh =', self.hardtanh)
        if self.hardtanh:
            sdff = sdff.to(self.torch_dtype['actf_int'])
        else :
            sdff = sdff.float() / (2 ** self.frac_bits_dict['sdff']) 
            
        act = self.one - self.tanh(sdff) ** 2
        # print(act)
        
        if self.hardtanh:
            act =torch.bitwise_right_shift(
                act, 
                self.frac_bits_dict['actf_int'] - self.frac_bits_dict['actf']
            ).to(self.torch_dtype['actf'])
        else :
            act = quantize_fixed_point(
                act, 
                self.frac_bits_dict['actf'], 
                *self.dtype_dict['actf']
            )
        # print(act)
        
        # == Linear layer with careful output scaling
        act_reshaped = act.view(batch_size, -1).to(self.torch_dtype['matmul'])
        weights = self.weight.to(self.torch_dtype['matmul'])     
        
        output = torch.nn.functional.linear(act_reshaped, weights)
        # print(weights, weights.min(), weights.max())
        # print(output, output.min(), output.max())
        output = torch.bitwise_right_shift(
            output,
            self.frac_bits_dict['matmul'] - self.frac_bits_dict['result']
        ).to(self.torch_dtype['result'])
        # print(output, output.min(), output.max())
        # exit()
        
        if (self.collect_stats):
            self.track_max = max(self.track_max, output.max())
            self.track_min = min(self.track_max, output.min())
        
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
    @overload
    def __init__(
        self, 
        model : FasterKAN,
        dtype_dict = default_dtype,
        frac_bits_dict = default_frac_bits,
        hardtanh = False,
        collect_stats = False
    ):
        ...
        
    @overload
    def __init__(
        self, layers_hidden, 
        num_grids: int = 1,
        dtype_dict = default_dtype,
        frac_bits_dict = default_frac_bits,
        hardtanh = False,
        collect_stats = False
    ):
        ...
    
    def __init__(
        self, 
        model : FasterKAN = None,
        layers_hidden = [], 
        num_grids: int = 1,
        dtype_dict = default_dtype,
        frac_bits_dict = default_frac_bits,
        hardtanh = False,
        collect_stats = False,
        **kwargs
    ):
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
        for qlayer, ulayer in zip(self.layers, fasterKAN.layers):
            qlayer.quantize(ulayer)
        
        return self
    
    def fit_frac_bits(self, x, fasterKAN: FasterKANLayer):
        output = x
        for qlayer, fasterKANLayer in zip(self.layers, fasterKAN.layers):
            output = qlayer.fit_frac_bits(output,fasterKANLayer)
        
        return self
    
    def fit_quantize(self, x, fasterKAN: FasterKAN):
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
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
     
    def update_hardtanh(self, hardtanh):
        for layer in self.layers:
            layer.update_hardtanh(hardtanh)
    
class FloatWrapperModule(nn.Module):
    def __init__(self, model : FixedPointFasterKAN):
        super(FloatWrapperModule, self).__init__()
        self.model = model
        
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
        
    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
        
    def forward(self,x):
        x = quantize_fixed_point(x, self.model.layers[0].frac_bits_dict['grid'], *self.model.layers[0].dtype_dict['grid'])
        return dequantize_fixed_point(self.model(x), self.model.layers[-1].frac_bits_dict['result'])
        
    def update_hardtanh(self,hardtanh=False):
        for layer in self.layers:
            layer.update_hardtanh(hardtanh)
        