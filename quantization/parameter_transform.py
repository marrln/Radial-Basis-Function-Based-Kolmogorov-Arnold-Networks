import re
import numpy as np
import torch
import os

def weight_transformer(
    weights: torch.Tensor,  
    RSLT_CHANNELS: int,
) -> torch.Tensor:
    # Extend with zeros
    if weights.shape[1] % RSLT_CHANNELS != 0:
        zeros = torch.zeros(weights.shape[0], RSLT_CHANNELS - weights.shape[1] % RSLT_CHANNELS).to(weights.dtype)
        weights = torch.cat([weights, zeros], dim=1)
    
    # Split the weights along columns into RSLT_CHANNELS chunks
    split_rslt = torch.split(weights, RSLT_CHANNELS, dim=1)
    # example_shape = split_rslt[0].shape
    # print(f"After splitting along columns: torch.Size({list(example_shape)}) * {len(split_rslt)}")
    # print(f"First split_rslt tensor:\n{split_rslt[0]}")
    
    # Concatenate along rows
    concat_rows = torch.cat(split_rslt, dim=0)
    # print(f"After concatenating along rows: {concat_rows.shape}")

    return concat_rows

def save_tensor_to_bin(tensor, fname):
    np_array = np.asarray(tensor)
    with open(fname, 'wb') as f:
        f.write(np_array.tobytes()) # default is 'C' order (row-major)
    print(f"Saved tensor to {fname}")

def save_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1):
    """
    Saves transformed linear layer weights to binary files.
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    weight_path = os.path.join(root_dir, "extracted_params",f"rslt_{RSLT_CHANNELS}")
    os.makedirs(weight_path, exist_ok=True)
    
    for key in state_dict:
        match = re.search(r'layers\.(\d+)', key)
        if not match:
            print(f"Key {key} does not contain layer index. Skipping.")
            continue
        layer_idx = match.group(1)
        
        # Extract weight tensor
        weight_tuple = state_dict[key]
        if isinstance(weight_tuple, tuple) :
            if len(weight_tuple) < 1:
                print(f"Invalid weight tuple for {key}. Skipping.")
                continue
            weight_tensor = weight_tuple[0]
        else :
            weight_tensor = weight_tuple
            
        if not isinstance(weight_tensor, torch.Tensor):
            print(f"Weight is not a tensor for {key}. Skipping.")
            continue

        # Convert quantized tensor to integer representation
        if weight_tensor.dtype == torch.qint8 and hasattr(weight_tensor, "int_repr"):
            weight_tensor = weight_tensor.int_repr().detach()
        else:
            weight_tensor = weight_tensor.detach().clone()
        
        # Apply transformation
        try:
            transformed = weight_transformer(weight_tensor.T, RSLT_CHANNELS)
        except Exception as e:
            print(f"Transformation failed for {key}: {e}")
            raise e
            continue
        
        if weight_tensor.dtype == torch.qint8 and hasattr(weight_tensor, "int_repr"):
            transformed = transformed.numpy().astype(np.int8)
        else:
            transformed = transformed.numpy()
            
        filename = f"layer_{layer_idx}_weight.bin"
        filepath = os.path.join(weight_path, filename)
        save_tensor_to_bin(transformed, filepath)

def extract_fx_packed_params(state_dict):
    return {key: value for key, value in state_dict.items() 
            if '_packed_params._packed_params' in key}

def save_fx_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1):
    """
    Saves transformed linear layer weights to binary files.
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    packed_params = extract_fx_packed_params(state_dict)
    return save_weights_to_bin(packed_params, root_dir, RSLT_CHANNELS)

def extract_custom_params(state_dict,parameter = 'weight'):
    return {key: value for key, value in state_dict['model_state_dict'].items() 
            if f'.{parameter}' in key}

def save_custom_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1):
    """
    Saves transformed linear layer weights to binary files.
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    packed_params = extract_custom_params(state_dict)
    return save_weights_to_bin(packed_params, root_dir, RSLT_CHANNELS)

def save_custom_model_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1):
    """
    Saves transformed linear layer weights to binary files.
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    save_custom_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS)
    param_dir = os.path.join(root_dir, "extracted_params")
    
    params = extract_custom_params(state_dict, 'grid')
    params.update(extract_custom_params(state_dict, 'inv_denom'))
    
    for key, val in params.items():
        save_tensor_to_bin(val, os.path.join(param_dir, f'{key.replace('.','_')}.bin'))
        
def packetize_model_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1):
    save_custom_model_to_bin(state_dict, root_dir, RSLT_CHANNELS)
    
    param_dir = os.path.join(root_dir, "extracted_params","")
    pckt_dir = os.path.join(root_dir, "packetized_params")
    
    os.makedirs(pckt_dir, exist_ok=True)
    
    for dirpath, dirnames, filenames in os.walk(param_dir):
        for key in ['weight','grid','inv_denom']:
            fnames = list(filter(lambda a: key in a, filenames))
            
            cumulative_fname = dirpath.replace(param_dir,'')
            cumulative_fname = f'{cumulative_fname}_{key}.bin' if len(cumulative_fname) else f'{key}.bin'
            cumulative_fname = os.path.join(pckt_dir, cumulative_fname)
            
            if len(fnames):
                with open(cumulative_fname, 'wb') as fw:
                    for fname in fnames:
                        with open(os.path.join(dirpath,fname), 'rb') as fr:
                            fw.write(fr.read())

                print(f"Packetized {key} to '{cumulative_fname}'")
            
if __name__ == "__main__":
    model_pth = '../Dataset/Custom-Quantizer/16_bits/45482/BCELoss/Adam/ReduceOnPlateau/3.0e-05/[12288,1024,7]/[4]/-2.0e+00/2.5e-01/2.0e+00/epoch_best/model_checkpoint.pth'
    # model_pth = '../Dataset/Custom-Quantizer/mixed_16_bits/45482/BCELoss/Adam/ReduceOnPlateau/3.0e-05/[12288,1024,7]/[4]/-2.0e+00/2.5e-01/2.0e+00/epoch_best/model_checkpoint.pth'
    
    os.chdir(os.path.dirname(__file__))
    if os.path.exists(model_pth):
        state_dict = torch.load(model_pth)
        root_dir = os.path.dirname(model_pth)
        RSLT_CHANNELS = 4
        
        packetize_model_to_bin(state_dict, root_dir, RSLT_CHANNELS)