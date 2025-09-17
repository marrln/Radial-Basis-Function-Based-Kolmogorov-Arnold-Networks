"""
This module provides utilities for managing model checkpoints and hyperparameter-based directory structures.

It includes functions for:
- Reading and validating configuration files.
- Creating checkpoint directories based on selected hyperparameters and config hashes.
- Saving and loading model and optimizer checkpoints.
- Collecting unique hyperparameter values from checkpoint directories.
- Processing model data and exporting results to Excel.

Functions:
- read_config():                        Read and validate a config.json file.
- get_checkpoint_dir():                 Construct a checkpoint directory using selected hyperparameters and a config hash.
- get_config_from_checkpoint():         Read config.json from a checkpoint directory.
- create_epoch_checkpoint_dir():        Create a directory for a specific epoch's checkpoint.
- save_model_checkpoint():              Save model and optimizer states, along with training metadata.
- load_model_checkpoint():              Load model and optimizer states from a checkpoint file.

Example usage:
    checkpoint_dir = get_checkpoint_dir(config, root_dir)
    save_model_checkpoint(epoch_dir, model, optimizer, epoch, loss)
    model, optimizer, start_epoch, loss, best_val_loss = load_model_checkpoint(model, device, model_path, optimizer_type, optimizer_params)
"""

import os
import json
import torch
import hashlib
from typing import Any, Dict, Optional

# Local imports
from . general_utils import load_json
from . import mapper 

# Settings
REQUIRED_KEYS = [
        "seed",
        "criterion",
        "optimizer",
        "dim_list",
        "grid_size_per_layer",
        "grid_min",
        "grid_max",
        "inv_denominator",
        "x_dim",
        "y_dim",
        "channel_size"
    ]


def read_config(config_path: str) -> Dict[str, Any]:
    """
    Reads and validates a config.json file, ensuring required keys are present.
    """
    # Accept either "learning_rate" or "learning_rate_stage1" as required
    if not ("learning_rate" in load_json(config_path) or "learning_rate_stage1" in load_json(config_path)):
        raise KeyError("Missing required config key: 'learning_rate' or 'learning_rate_stage1'")
    config = load_json(config_path)
    missing_keys = [k for k in REQUIRED_KEYS if k not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")
    
    # check that dim_list[0] == x_dim * y_dim * channel_size
    if config['dim_list'][0] != config['x_dim'] * config['y_dim'] * config['channel_size']:
        raise ValueError(f"dim_list[0] ({config['dim_list'][0]}) must equal x_dim*y_dim*channel_size ({config['x_dim']*config['y_dim']*config['channel_size']})")
    # check that dim_list[-1] == output_classes if output_classes is not None
    if config.get('output_classes') is not None and config['dim_list'][-1] != config['output_classes']:
        raise ValueError(f"dim_list[-1] ({config['dim_list'][-1]}) must equal output_classes ({config['output_classes']})")
    
    return config


def get_checkpoint_dir(config: Dict[str, Any], root_dir: str) -> str:
    """
    Constructs a checkpoint directory using selected human-readable hyperparameters and a hash of the full config.
    Only includes important hyperparameters in the directory name, omitting those that are not relevant for uniqueness.
    Adds a hash for full config uniqueness.
    """
    readable_keys = [
        "criterion",
        "optimizer",
        "scheduler" if config.get("scheduler") else None,
        "learning_rate_stage1" if config.get("learning_rate_stage1") else "learning_rate", # NOTE: for HAM10000 experiments, because training was done in two stages
        "grid_size_per_layer",
    ]

    # NOTE: the following is for HAM10000 experiments
    if config.get("exclude_mnv_in_stage1"):
        readable_keys.append("exclude_mnv_in_stage1")
    
    readable_keys = [k for k in readable_keys if k]
    readable_values = []
    for k in readable_keys:
        v = config.get(k)
        if isinstance(v, (list, dict)):
            v_str = json.dumps(v)
        else:
            v_str = str(v)
        readable_values.append(v_str)

    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8] # Shorten hash for readability
    dir_name = "_".join(readable_values) + f"_{config_hash}"
    checkpoint_dir = os.path.join(root_dir, dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save a copy of config.json in the checkpoint directory
    config_copy_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_copy_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return checkpoint_dir


def get_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Reads config.json from the given checkpoint directory.
    If config.json is not found, raises a FileNotFoundError.

    Args:
        checkpoint_path (str): Path to the checkpoint directory.

    Returns:
        Dict[str, Any]: Dictionary containing the configuration.
    """
    config_file = os.path.join(checkpoint_path, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"config.json not found in checkpoint directory: {checkpoint_path}")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_epoch_checkpoint_dir(epoch: int, checkpoint_dir: str) -> str:
    """
    Create a directory for the current epoch's checkpoint.
    """
    epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    return epoch_dir


def save_model_checkpoint(
    epoch_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    best_val_loss: Optional[float] = None,
    optimizer_init_params: Optional[dict] = None
) -> None:
    """
    Save the model and optimizer states, along with training metadata, to a checkpoint file.
    Optionally saves the optimizer's initialization parameters for reproducibility.

    Args:
        epoch_dir (str): Directory to save the checkpoint.
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        epoch (int): Current epoch.
        loss (float): Current loss.
        best_val_loss (Optional[float]): Best validation loss so far.
        optimizer_init_params (Optional[dict]): Optimizer initialization parameters to save (optional).
    """
    model_path = os.path.join(epoch_dir, 'model_checkpoint.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss
    if optimizer_init_params is not None:
        checkpoint['optimizer_init_params'] = optimizer_init_params
    torch.save(checkpoint, model_path)


def load_model_checkpoint(
    model: torch.nn.Module,
    device: str,
    model_path: str = 'invalid_path',
    optimizer_type: str = None,
    optimizer_params: dict = None
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float, Optional[float]]:
    """
    Load the checkpoint for a model and optionally an optimizer if you want to continue the training.
    """
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only = True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        optimizer: Optional[torch.optim.Optimizer] = None
        if optimizer_type:
            optimizer_params = optimizer_params or {}
            optimizer = mapper.get_optimizer(optimizer_type, model.parameters(), optimizer_params)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch: int = checkpoint.get('epoch', 0) + 1
        loss: float = checkpoint.get('loss', float('inf'))
        best_val_loss: Optional[float] = checkpoint.get('best_val_loss', None)
        print(f"Checkpoint loaded from {model_path}. Next epoch is {start_epoch}.")
        return model, optimizer, start_epoch, loss, best_val_loss
    else:
        raise FileNotFoundError(f"Checkpoint file not found at path: {model_path}")