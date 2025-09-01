"""

This module provides utilities for managing model checkpoints and hyperparameter-based directory structures.

It includes functions for:
- Building and parsing hyperparameter dictionaries.
- Creating checkpoint directories based on hyperparameter values.
- Saving and loading model and optimizer checkpoints.
- Collecting unique hyperparameter values from checkpoint directories.
- Counting model parameters and calculating MACs (Multiply-Accumulate Operations).
- Saving model attributes and details to a text file.
- Processing model data from directories and saving it to an Excel file.

Functions:
- build_hyperparams_typedict():           Build a dictionary mapping hyperparameter names to their types.
- make_checkpoint_dir_from_hyperparams(): Construct a directory path encoding hyperparameter values.
- parse_hyperparams_from_dirname():       Parse hyperparameter values from a directory path.
- create_epoch_checkpoint_dir():          Create a directory for a specific epoch's checkpoint.
- save_model_checkpoint():                Save model and optimizer states, along with training metadata.
- load_model_checkpoint():                Load model and optimizer states from a checkpoint file.
- collect_unique_hyperparams_from_dirs(): Collect all unique hyperparameter values from checkpoint directories.\
- load_json():                            Load a JSON file.
- count_parameters():                     Count total and trainable parameters in a model.
- get_model_macs_params():                Calculate MACs, total parameters, and trainable parameters for a model.
- save_attributes():                      Save model attributes and details to a text file.
- process_model_data():                   Process model data from directories and save it to an Excel file.

example_hyperparams_typedict: Dict[str, type] = {
    "seed": int,
    "criterion": str,
    "optimizer": str,
    "scheduler": str,
    "dim_list": list,
    "learning_rate": float,
    "grid_size_per_layer": list,
    "grid_min": float,
    "grid_max": float,
    "inv_denominator": float,
}

This results in a directory structure like:
root_dir/42/CrossEntropy/Adam/StepLR/[12288,256,10]/0.001/[4,8,16]/-1.2/0.25/0.5

"""


import os
import json
import torch
import mapper 
import pandas as pd
from torchinfo import summary
from thop import profile  
from itertools import product
from typing import Any, Dict, Optional, Tuple


def build_hyperparams_typedict(**kwargs) -> Dict[str, type]:
    """
    Build a dictionary of model hyperparameters just like the example_hyperparams_typedict.
    Returns:
        Dict[str, type]: Dictionary of hyperparameters.
    """
    hyperparams_typedict: Dict[str, type] = {}
    for key in kwargs:
        hyperparams_typedict[key] = kwargs[key]
    return hyperparams_typedict


def make_checkpoint_dir_from_hyperparams(hyperparams_typedict: Dict[str, type], hyperparams: Dict[str, Any], root_dir: str) -> str:
    """
    Constructs a directory path for storing model checkpoints based on the provided hyperparams_typedict.
    The directory format will be .../value1/value2/.../valueN, where each value corresponds to a hyperparameter value (not name or type),
    in the order of hyperparams_typedict keys.

    Args:
        hyperparams_typedict (Dict[str, type]): A dictionary of hyperparameter types, where keys are parameter names and values are their types.
        hyperparams (Dict[str, Any]): A dictionary of hyperparameter values to include in the directory path.
        root_dir (str): The root directory under which the checkpoint directory will be created.

    Returns:
        str: The constructed directory path, which encodes the hyperparameters' values as nested directories.

    Example:
        >>> hyperparams = {'lr': 0.01, 'batch_size': 32, 'activation': 'relu'}
        >>> root_dir = '/checkpoints'
        >>> make_checkpoint_dir_from_hyperparams(hyperparams_typedict, hyperparams, root_dir)
        '/checkpoints/0.01/32/relu'
    """
    try:
        if set(hyperparams_typedict.keys()) != set(hyperparams.keys()):
            raise ValueError("Keys of hyperparams_typedict and hyperparams must match exactly.")
        hp_values = []
        for k in hyperparams_typedict:
            typ = hyperparams_typedict[k]
            v = hyperparams[k]
            # Force type
            if typ is list and not isinstance(v, list):
                v = [v]
            elif typ is not list and not isinstance(v, typ):
                v = typ(v)
            # Serialize lists and dicts as JSON strings
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v)
            else:
                v_str = str(v)
            hp_values.append(v_str)
        checkpoint_dir = os.path.join(root_dir, *hp_values)
        return checkpoint_dir
    except Exception as e:
        raise ValueError(f"Error constructing checkpoint directory: {e}")


def parse_hyperparams_from_dirname(dir_path: str, hyperparams_typedict: Dict[str, type]) -> Dict[str, Any]:
    """
    Deconstructs a directory path into its constituent hyperparameter values, using the provided hyperparams dict for types.
    This function expects the directory name to be formatted as value1/value2/.../valueN, where each value corresponds to a hyperparameter.

    Args:
        dir_path (str): The directory path containing hyperparameter values in its name.
        hyperparams_typedict (Dict[str, type]): A dictionary where keys are hyperparameter names and values are their expected types.

    Returns:
        Dict[str, Any]: Dictionary of hyperparameter names and their parsed values.
    """
    try:
        # Split the directory path and get the last N parts, where N = number of hyperparameters
        dir_parts = os.path.normpath(dir_path).split(os.sep)
        num_hyperparams = len(hyperparams_typedict)
        if len(dir_parts) < num_hyperparams:
            raise ValueError(
                f"Directory path '{dir_path}' does not contain enough parts for {num_hyperparams} hyperparameters."
            )
        # Take the last N parts as the hyperparameter values
        hp_values = dir_parts[-num_hyperparams:]
        parsed: Dict[str, Any] = {}
        for (k, typ), v in zip(hyperparams_typedict.items(), hp_values):
            # Try to parse JSON for complex types, else cast directly
            try:
                value = json.loads(v)
            except Exception:
                value = v
            # Convert to the expected type if not already
            if typ is list and not isinstance(value, list):
                value = [value]
            elif typ is not list and not isinstance(value, typ):
                value = typ(value)
            parsed[k] = value
        return parsed
    except Exception as e:
        raise ValueError(f"Error deconstructing directory path: {e}")


def create_epoch_checkpoint_dir(epoch: int, checkpoint_dir: str) -> str:
    """
    Create a directory for the current epoch's checkpoint.
    """
    epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    # Optionally provide feedback about directory creation
    if not os.path.exists(epoch_dir):
        print(f"Created new directory: {epoch_dir}")
    else:
        print(f"Directory already exists: {epoch_dir}")
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
    checkpoint_path = os.path.join(epoch_dir, 'model_checkpoint.pth')
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
    torch.save(checkpoint, checkpoint_path)


def load_model_checkpoint(
    model: torch.nn.Module,
    device: str,
    checkpoint_path: str = 'invalid_path',
    optimizer_type: str = None,
    optimizer_params: dict = None
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float, Optional[float]]:
    """
    Load the checkpoint for a model and optionally an optimizer.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
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
        print(f"Checkpoint loaded from {checkpoint_path}. Next epoch is {start_epoch}.")
        return model, optimizer, start_epoch, loss, best_val_loss
    else:
        raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint_path}")


def collect_unique_hyperparams_from_dirs(root_dir: str, hyperparams_typedict: Dict[str, type]) -> Dict[str, list]:
    """
    Extracts all unique hyperparameter values from directory names using the provided hyperparams_typedict keys and types.
    This function scans through all subdirectories in the specified root directory, expecting each directory name to 
    be formatted as value1/value2/.../valueN, where each value corresponds to a hyperparameter.

    Args:
        root_dir (str): The root directory containing checkpoint subdirectories.
        hyperparams_typedict (Dict[str, type]): Dictionary of hyperparameter names and their types.

    Returns:
        Dict[str, list]: Dictionary mapping each hyperparameter name to a sorted list of unique values found.
    """
    unique_hyperparams: Dict[str, set] = {key: set() for key in hyperparams_typedict.keys()}
    num_hyperparams = len(hyperparams_typedict)
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        # Split the directory path and get the last N parts, where N = number of hyperparameters
        dir_parts = os.path.normpath(dir_path).split(os.sep)
        if len(dir_parts) < num_hyperparams:
            continue
        hp_values = dir_parts[-num_hyperparams:]
        hp = {}
        try:
            for (k, typ), v in zip(hyperparams_typedict.items(), hp_values):
                try:
                    value = json.loads(v)
                except Exception:
                    value = v
                if typ is list and not isinstance(value, list):
                    value = [value]
                elif typ is not list and not isinstance(value, typ):
                    value = typ(value)
                hp[k] = value
            for key in hyperparams_typedict.keys():
                if key in hp:
                    value = tuple(hp[key]) if isinstance(hp[key], list) else hp[key]
                    unique_hyperparams[key].add(value)
        except Exception as e:
            print(
                f"Error processing directory '{dir_path}': {e}\n"
                f"Expected directory path format: .../value1/value2/.../valueN where N = {num_hyperparams}."
            )
            continue

    result: Dict[str, list] = {}
    for key, values in unique_hyperparams.items():
        processed = []
        for v in values:
            if isinstance(v, tuple):
                processed.append(list(v))
            else:
                processed.append(v)
        result[key] = sorted(processed, key=lambda x: str(x))
    return result


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Counts the total and trainable parameters in a PyTorch model.
    Args:
        model (torch.nn.Module): The PyTorch model.
    Returns:
        tuple: A tuple containing:
            - total_params (int): The total number of parameters in the model.
            - trainable_params (int): The number of trainable parameters in the model.
    """

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_model_macs_params(
    model: torch.nn.Module, 
    x_dim: int, 
    y_dim: int, 
    channel_size: int
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Calculates the MACs (Multiply-Accumulate Operations), total parameters, and trainable parameters for a given model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        x_dim (int): The width of the input image.
        y_dim (int): The height of the input image.
        channel_size (int): The number of channels in the input image.

    Returns:
        tuple: (macs, total_params, trainable_params) or (None, None, None) if calculation fails.
    """
    total_params, trainable_params = count_parameters(model)
    try:
        input_tensor = torch.randn(1, channel_size * x_dim * y_dim)
        macs, _ = profile(model.cpu(), inputs=(input_tensor,))
    except Exception as e:
        print(f"Error calculating MACs or parameters: {e}")
        return None, None, None
    return macs, total_params, trainable_params


def save_attributes(
    model: torch.nn.Module,
    root_dir: str,
    hyperparams: dict,
    hyperparams_typedict: dict,
    x_dim: int,
    y_dim: int,
    channel_size: int,
    optimizer_init_params: dict = None,
    scheduler_init_params: dict = None,
    criterion_init_params: dict = None
) -> str:
    """
    Saves the attributes and details of a PyTorch model to a text file, including
    its parameter counts, architecture summary, and MACs (Multiply-Accumulate Operations).
    Also creates a checkpoint directory using the updated checkpoint module logic.

    Args:
        model (torch.nn.Module): The PyTorch model.
        root_dir (str): The root directory where model checkpoint folders are located.
        hyperparams (dict): Dictionary of hyperparameters (must include 'dim_list', etc.).
        hyperparams_typedict (dict): Dictionary mapping hyperparameter names to types.
        x_dim (int): The width of the input image.
        y_dim (int): The height of the input image.
        channel_size (int): The number of channels in the input image.
        optimizer_init_params (dict, optional): Optimizer initialization parameters to save (optional).

    Returns:
        str: The directory path where the attributes were saved.
    """

    try:
        dimension_list = hyperparams['dim_list']
    except KeyError:
        raise KeyError("The model's hyperparameter dictionary must contain a key named 'dim_list'. Please double check your hyperparameters and ensure the dimensions are named as 'dim_list'.")
    input_size = (1, channel_size * x_dim * y_dim)
    if dimension_list[0] != channel_size * x_dim * y_dim:
        raise ValueError(f"The first dimension of the dimension list must match the input size. Got {dimension_list[0]} but expected {channel_size * x_dim * y_dim}.")

    macs, total_params, trainable_params = get_model_macs_params(model, x_dim, y_dim, channel_size)

    dir_path = make_checkpoint_dir_from_hyperparams(hyperparams_typedict, hyperparams, root_dir)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, "model_summary.txt")
    with open(file_path, "w", encoding='utf-8') as file:
        file.write("Model Hyperparameters:\n")
        if 'dim_list' in hyperparams:
            file.write(f"\nDimension List: {hyperparams['dim_list']}\n")
            file.write(f"Number of Layers: {len(hyperparams['dim_list']) - 1}\n\n")
        for key, value in hyperparams.items():
            if key == 'dim_list':
                continue
            file.write(f"{key}: {value}\n")
            if key == 'optimizer' and optimizer_init_params:
                file.write(f"\n  Optimizer Params: {optimizer_init_params}\n")
            if key == 'scheduler' and scheduler_init_params:
                file.write(f"\n  Scheduler Params: {scheduler_init_params}\n")
            if key == 'criterion' and criterion_init_params:
                file.write(f"\n  Criterion Params: {criterion_init_params}\n")
        file.write(f"\nTotal Parameters: {total_params}\n")
        file.write(f"Trainable Parameters: {trainable_params}\n")
        file.write(f"MACs (Multiply-Accumulate Operations): {macs}\n")
        file.write("\nModel Summary:\n")
        file.write(str(summary(model, input_size=input_size)))

    return dir_path


def process_model_data(
    root_dir: str,
    hyperparams_typedict: dict
) -> None:
    """
    Processes model data from directories, extracts relevant information, and saves it to an Excel file.

    Args:
        root_dir (str): The root directory where model checkpoint folders are located.
        hyperparams_typedict (dict): Dictionary mapping hyperparameter names to types.
    """

    hyperparams_values = collect_unique_hyperparams_from_dirs(root_dir, hyperparams_typedict)
    hparam_names = list(hyperparams_typedict.keys())
    value_lists = [hyperparams_values[name] for name in hparam_names]

    data = []
    for values in product(*value_lists):
        hyperparams = dict(zip(hparam_names, values))
        model_dir = make_checkpoint_dir_from_hyperparams(hyperparams_typedict, hyperparams, root_dir)

        files = {
            "attributes": os.path.join(model_dir, "model_summary.txt"),
            "accuracy": os.path.join(model_dir, "accuracy_logs.json"),
            "loss": os.path.join(model_dir, "loss_logs.json"),
        }

        if not all(os.path.isfile(f) for f in files.values()):
            print(f"Files missing in {model_dir}. Skipping.")
            continue

        acc = load_json(files["accuracy"])
        loss = load_json(files["loss"])

        epochs = zip(
            acc.get("training_accuracy", []),
            acc.get("validation_accuracy", []),
            loss.get("training_loss", []),
            loss.get("validation_loss", []),
        )

        for epoch_idx, (tr_acc, val_acc, tr_loss, val_loss) in enumerate(epochs, 1):
            data.append({
                "Epoch": epoch_idx,
                "Train Acc": tr_acc,
                "Val Acc": val_acc,
                "Train Loss": tr_loss,
                "Val Loss": val_loss,
                "Dir": model_dir,
            })

    df = pd.DataFrame(data)
    output_file = "model_summary_final.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Data has been saved to {output_file}")
