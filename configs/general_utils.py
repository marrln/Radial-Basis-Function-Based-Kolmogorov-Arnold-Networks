import json
import torch
from thop import profile  
from typing import Optional, Tuple

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Counts the total and trainable parameters in a PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_model_macs_params(model: torch.nn.Module,  config: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Calculates the MACs (Multiply-Accumulate Operations), total parameters, 
    and trainable parameters for a given model,
    using x_dim, y_dim, and channel_size from the config dictionary.

    Args:
        model (torch.nn.Module): The PyTorch model.
        config (dict): Configuration dictionary containing 'x_dim', 'y_dim', and 'channel_size'.

    Returns:
        tuple: (macs, total_params, trainable_params) or (None, None, None) if calculation fails.
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
        return None, None, None
    return macs, total_params, trainable_params