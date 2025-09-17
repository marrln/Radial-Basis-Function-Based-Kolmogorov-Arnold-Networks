"""
Experiment Evaluation Utilities for RBF-based Kolmogorov-Arnold Networks (KANs)

This module provides tools for evaluating and comparing RBF-KAN experiments, including:
1. Model attribute recording and summarization
2. Extraction of training/validation metrics across experiments
3. Collection and aggregation of hyperparameters from multiple experiment runs
4. Data processing for comparative analysis of different model configurations

Key Features:
- Save detailed model attributes including architecture, parameters count, and MACs
- Extract and collect unique hyperparameter values across multiple experiments
- Process and consolidate training metrics from multiple experiment directories
- Export aggregated data to Excel for further analysis and visualization

Example Usage:
    # Save model attributes to a checkpoint directory
    checkpoint_path = save_attributes(
        model=model,
        root_dir="experiments",
        config=config_dict
    )
    
    # Collect unique hyperparameter values from all experiments
    unique_params = collect_unique_hyperparams_from_dirs("experiments")
    
    # Process all experiment data and export to Excel
    process_model_data("experiments", config_dict)

This module works in conjunction with the checkpoint_utils and general_utils modules
to provide a comprehensive experiment management and evaluation framework for RBF-KANs.

"""

import os
import json
import pandas as pd
import torch
from itertools import product
from torchinfo import summary
from typing import Dict

# Local imports
from . import general_utils as utils
from . import checkpoint_utils as checkpoint

def save_attributes(
    model: torch.nn.Module,
    root_dir: str,
    config: dict,
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
        config (dict): Dictionary of hyperparameters (must include 'dim_list', etc.).
        optimizer_init_params (dict, optional): Optimizer initialization parameters to save (optional).
        scheduler_init_params (dict, optional): Scheduler initialization parameters to save (optional).
        criterion_init_params (dict, optional): Criterion initialization parameters to save (optional).

    Returns:
        str: The directory path where the attributes were saved.
    """

    x_dim = config.get('x_dim')
    y_dim = config.get('y_dim')
    channel_size = config.get('channel_size')

    macs, total_params, trainable_params = utils.get_model_macs_params(model, config)

    checkpoint_path = checkpoint.get_checkpoint_dir(config, root_dir)
    os.makedirs(checkpoint_path, exist_ok=True)

    file_path = os.path.join(checkpoint_path, "model_summary.txt")
    with open(file_path, "w", encoding='utf-8') as file:
        file.write("Model Hyperparameters:\n")
        if 'dim_list' in config:
            file.write(f"\nDimension List: {config['dim_list']}\n")
            file.write(f"Number of Layers: {len(config['dim_list']) - 1}\n\n")
        for key, value in config.items():
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
        file.write(str(summary(model, input_size=(1, x_dim * y_dim * channel_size))))

    return checkpoint_path

def collect_unique_hyperparams_from_dirs(root_dir: str) -> Dict[str, list]:
    """
    Walks through the root directory, collects all config.json files, and finds unique values for all keys across them.
    Args:
        root_dir (str): The root directory containing checkpoint subdirectories.
    Returns:
        Dict[str, list]: Dictionary mapping each config key to a sorted list of unique values found.
    """
    unique_hyperparams: Dict[str, set] = {}

    for dirpath, _, filenames in os.walk(root_dir):
        if "config.json" in filenames:
            config_path = os.path.join(dirpath, "config.json")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                for key, value in config.items():
                    if key not in unique_hyperparams:
                        unique_hyperparams[key] = set()
                    # Convert lists to tuples for set hashing
                    if isinstance(value, list):
                        unique_hyperparams[key].add(tuple(value))
                    else:
                        unique_hyperparams[key].add(value)
            except Exception as e:
                print(f"Error reading {config_path}: {e}")

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


def process_model_data(root_dir: str, config: dict) -> None:
    """
    Processes model data from directories, extracts relevant information, and saves it to an Excel file.
    Args:
        root_dir (str): The root directory where model checkpoint folders are located.
        hyperparams_typedict (dict): Dictionary mapping hyperparameter names to types.
    """

    hyperparams_values = collect_unique_hyperparams_from_dirs(root_dir)
    hparam_names = list(config.keys())
    value_lists = [hyperparams_values[name] for name in hparam_names]

    data = []
    for values in product(*value_lists):
        hyperparams = dict(zip(hparam_names, values))
        model_dir = checkpoint.get_checkpoint_dir(config, root_dir)

        files = {
            "accuracy": os.path.join(model_dir, "accuracy_logs.json"),
            "loss": os.path.join(model_dir, "loss_logs.json"),
        }

        if not all(os.path.isfile(f) for f in files.values()):
            print(f"Files missing in {model_dir}. Skipping.")
            continue

        acc = utils.load_json(files["accuracy"])
        loss = utils.load_json(files["loss"])

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
