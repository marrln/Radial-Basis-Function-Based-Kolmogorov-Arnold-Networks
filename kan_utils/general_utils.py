"""
General utility functions for RBF-KAN models.

This module provides helper functions for common tasks such as:
- Loading JSON configuration files
- Counting model parameters
- Calculating computational complexity (MACs)

These utilities support the training, evaluation, and analysis of 
Radial Basis Function-based Kolmogorov-Arnold Networks.
"""

import json
import torch
from thop import profile  
from typing import Optional, Tuple, Dict, Any

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file to load
        
    Returns:
        Dict[str, Any]: Parsed JSON content as a dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Counts the total and trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model to analyze
        
    Returns:
        tuple[int, int]: A tuple containing (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_model_macs_params(model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[Optional[int], int, int]:
    """
    Calculates the MACs (Multiply-Accumulate Operations), total parameters, 
    and trainable parameters for a given model,
    using x_dim, y_dim, and channel_size from the config dictionary.

    Args:
        model (torch.nn.Module): The PyTorch model.
        config (dict): Configuration dictionary containing 'x_dim', 'y_dim', and 'channel_size'.

    Returns:
        Tuple[Optional[int], int, int]: A tuple containing:
            - macs: Number of MACs (or None if calculation fails)
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
    """
    x_dim = config.get('x_dim')
    y_dim = config.get('y_dim')
    channel_size = config.get('channel_size')
    total_params, trainable_params = count_parameters(model)
    try:
        input_tensor = torch.randn(1, channel_size * x_dim * y_dim)
        macs, _ = profile(model.cpu(), inputs=(input_tensor,))
    except Exception as e:
        print(f"Error calculating MACs or parameters: {e}")
        return None, total_params, trainable_params
    return macs, total_params, trainable_params