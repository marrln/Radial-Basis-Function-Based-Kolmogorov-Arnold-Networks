
import torch
import numpy as np
import os

def get_grids(checkpoint_path, device=torch.device('cpu')):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict']
    grids = {}
    for key, value in state_dict.items():
        if "grid" in key:
            grids[key] = value
    return grids

def get_inv_denoms(checkpoint_path, device=torch.device('cpu')):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict']
    inv_denoms = {}
    for key, value in state_dict.items():
        if "inv_denom" in key:
            inv_denoms[key] = value
    return inv_denoms

def get_linear_weights(checkpoint_path, device=torch.device('cpu')):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict']
    linear_weights = {}
    for key, value in state_dict.items():
        if "weight" in key:
            linear_weights[key] = value
    return linear_weights


def transform_linear_weights(linear_weights: torch.Tensor, RSLT_channels: int) -> torch.Tensor:
    """
    Rearranges a linear layer's weight matrix for efficient hardware inference.

    Given a weight matrix W of shape (M, N):
    1. If N is not divisible by RSLT_channels, pad columns with zeros.
    2. Split W into submatrices along columns, each with RSLT_channels columns.
    3. Stack these submatrices vertically to form a new matrix of shape (M * (N // RSLT_channels), RSLT_channels).

    Args:
        linear_weights (torch.Tensor): Weight matrix of shape (M, N).
        RSLT_channels (int): Number of output channels processed in parallel by the hardware.

    Returns:
        torch.Tensor: Rearranged weight matrix of shape (M * ceil(N / RSLT_channels), RSLT_channels).
    """
    rows, cols = linear_weights.shape

    # Pad columns if necessary
    if cols % RSLT_channels != 0:
        pad_cols = RSLT_channels - (cols % RSLT_channels)
        padding = torch.zeros((rows, pad_cols), dtype=linear_weights.dtype, device=linear_weights.device)
        linear_weights = torch.cat([linear_weights, padding], dim=1)
        cols += pad_cols

    # Split and stack
    split_matrices = torch.split(linear_weights, RSLT_channels, dim=1)
    transformed = torch.cat(split_matrices, dim=0)

    return transformed


def save_to_bin(tensor: torch.Tensor, file_path: str, dtype=np.int16):
    """
    Saves a PyTorch tensor to a binary file with a specified dtype.

    Args:
        tensor (torch.Tensor): The tensor to save.
        file_path (str): Path to save the binary file.
        dtype: Numpy dtype to use for saving the tensor. Default is np.int16.
    """
    np_array = tensor.cpu().numpy().astype(dtype)
    np_array.tofile(file_path)
    # print(f"Saved tensor to {file_path} with dtype {dtype}")


def save_linear_weights(linear_weights: torch.Tensor, file_path: str, C: int, dtype=np.int16):
    """
    Saves the transformed linear weights to a binary file after transforming them to row-major order.
    
    Args:
        linear_weights (torch.Tensor): The original weights tensor.
        file_path (str): Path to save the binary file.
        C (int): Number of results the acceleration device can infer in a single iteration.
    """
    # transform the weights
    transformed_weights = transform_linear_weights(linear_weights, RSLT_channels=C) 
    save_to_bin(transformed_weights, file_path, dtype=dtype)


if __name__ == "__main__":
    checkpoint_dir = r"Custom-Quantizer\16_bits\45482\BCELoss\Adam\ReduceOnPlateau\3.0e-05\[12288,1024,7]\[4]\-2.0e+00\2.5e-01\2.0e+00\epoch_best"
    pth_file = os.path.join(checkpoint_dir, "model_checkpoint.pth")  # Ensure the full path to the .pth file
    folder_pth = r"C:\Users\mrlnp\OneDrive - National and Kapodistrian University of Athens\Υπολογιστής\KANs\Custom Quantized Models"
    pth_file = os.path.join(folder_pth, pth_file)  

    # Create output directory for binary files
    output_dir = os.path.join(folder_pth, checkpoint_dir)

    grids = get_grids(pth_file)
    # print("Grids:", grids)
    inv_denoms = get_inv_denoms(pth_file)
    # print("Inv Denoms:", inv_denoms)
    linear_weights = get_linear_weights(pth_file)
    # print("Linear Weights:", linear_weights)

    # save all grids to binary files
    for key, value in grids.items():
        file_name = f"{key.replace('.', '_')}.bin"  
        file_path = os.path.join(output_dir, file_name)
        save_to_bin(value, file_path, dtype=np.int16) 

    # save all inv_denoms to binary files
    for key, value in inv_denoms.items():
        file_name = f"{key.replace('.', '_')}.bin"  
        file_path = os.path.join(output_dir, file_name)
        save_to_bin(value, file_path, dtype=np.int16)  

    # save all linear weights to binary files
    C = 8  
    for key, value in linear_weights.items():
        file_name = f"{key.replace('.', '_')}_transformed.bin"  
        file_path = os.path.join(output_dir, file_name)
        save_linear_weights(value, file_path, C, dtype=np.int16)