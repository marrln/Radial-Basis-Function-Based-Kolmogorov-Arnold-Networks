"""
Image Classification Training Pipeline for RBF-based Kolmogorov-Arnold Networks (KANs)

This module provides a comprehensive training pipeline for image classification tasks
using Radial Basis Function (RBF) based Kolmogorov-Arnold Networks. It includes functions
for model initialization, training, validation, checkpointing, and performance metrics computation.

The module is designed to work with the FasterKAN implementation, which is an optimized version
of RBF-KANs with features like dropout scaling and gradient boosting.

Key Features:
- Model initialization from configuration files
- End-to-end training with validation
- Checkpointing and model serialization
- Training/validation metrics logging and visualization
- Detailed performance metrics computation (accuracy, F1 score, recall, confusion matrix)
- Support for early stopping and learning rate scheduling

Example Usage:
    # Initialize model from config
    model, attr_path = initialize_kan_model_from_config('config.json', device='cuda')
    
    # Train and validate the model
    train_and_validate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion='CrossEntropyLoss',
        optimizer='Adam',
        scheduler='ReduceLROnPlateau',
        device='cuda',
        checkpoint_dir='checkpoints',
        epochs=50,
        patience=10
    )
    
    # Validate a trained model
    val_loss, val_accuracy = validate_model(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        checkpoint_path='checkpoints/best/model_checkpoint.pth',
        device='cuda',
        metrics_flag=True
    )
"""

import os
import re
import json
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from typing import Dict, List, Tuple, Union, Optional, Any

# Local imports
from . import checkpoint_utils as checkpoint
from . import experiment_eval as exeval
from . import fasterkan
from . import mapper

# Settings
SAVE_METRICS_IN_TXT = True
CHECKPOINT_DIRECTORY = "Training Checkpoints"


def initialize_kan_model_from_config(config_path: str, device: str = 'cpu') -> tuple[torch.nn.Module, str]:
    """
    Initialize the FasterKAN model using a configuration file path.
    
    Args:
        config_path (str): Path to the config.json file containing model hyperparameters.
        device (str): Device to use for the model ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple[torch.nn.Module, str]: 
            - model: Initialized FasterKAN model.
            - file_path: Path to the saved model attributes file.
    
    Raises:
        KeyError: If required configuration keys are missing.
        ValueError: If model dimensions are inconsistent.
    """
    
    config = checkpoint.read_config(config_path)    

    model = fasterkan.FasterKAN(
        layers_hidden=config['dim_list'],
        num_grids=config['grid_size_per_layer'],
        grid_min=config['grid_min'],
        grid_max=config['grid_max'],
        inv_denominator=config['inv_denominator']
    ).to(device)

    file_path = exeval.save_attributes(
        model=model,
        root_dir=CHECKPOINT_DIRECTORY,
        config=config
    )
    return model, file_path


def update_logs(log_file: str, epoch: int, training_metric: float, validation_metric: float, metric_name: str) -> None:
    """
    Update log files for training and validation metrics in JSON format.

    Args:
        log_file (str): Path to the log file.
        epoch (int): Current epoch number.
        training_metric (float): Metric value for training (e.g., loss or accuracy).
        validation_metric (float): Metric value for validation (e.g., loss or accuracy).
        metric_name (str): Name of the metric (e.g., "Loss", "Accuracy").
    
    Returns:
        None
    
    Note:
        If the log file already exists, it will be updated. If an entry for the current epoch
        and metric already exists, it will be replaced with the new values.
    """
    # Load existing logs if file exists
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    # Remove any existing entry for this epoch and metric_name
    logs = [entry for entry in logs if not (entry.get("epoch") == epoch and entry.get("metric_name") == metric_name)]

    # Append new log entry
    logs.append({
        "epoch": epoch,
        f"training_{metric_name}": training_metric,
        f"validation_{metric_name}": validation_metric
    })

    # Save logs back to file
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

def compute_and_serialize_metrics(
    all_targets: torch.Tensor,
    all_preds: torch.Tensor,
    val_accuracy: float,
    checkpoint_path: Optional[str] = None,
    save_txt: bool = SAVE_METRICS_IN_TXT
) -> Dict[str, Any]:
    """
    Compute F1, recall, confusion matrix, and accuracy, print them, and save to JSON and optionally TXT files.

    Args:
        all_targets (torch.Tensor): True labels tensor.
        all_preds (torch.Tensor): Predicted labels tensor.
        val_accuracy (float): Validation accuracy as a percentage.
        checkpoint_path (Optional[str]): Path to checkpoint for directory context. Defaults to None.
        save_txt (bool): Whether to also save metrics as TXT. Defaults to SAVE_METRICS_IN_TXT.

    Returns:
        Dict[str, Any]: Dictionary containing computed metrics (f1_score, recall, accuracy, confusion_matrix).
    
    Notes:
        - For multi-class classification, 'weighted' averages metrics by support 
          (number of true instances for each label).
        - 'macro' would give equal weight to each class regardless of support.
        - We use 'weighted' for imbalanced classes to reflect the class distribution.
    """

    f1 = f1_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)

    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {val_accuracy:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "f1_score": float(f1),
        "recall": float(recall),
        "accuracy": float(val_accuracy),
        "confusion_matrix": cm.tolist()
    }

    if checkpoint_path:
        metrics_dir = os.path.dirname(checkpoint_path)
        metrics_json_path = os.path.join(metrics_dir, "metrics.json")
        with open(metrics_json_path, 'w') as jf:
            json.dump(metrics, jf, indent=2)
        if save_txt:
            metrics_txt_path = os.path.join(metrics_dir, "metrics.txt")
            with open(metrics_txt_path, 'w') as tf:
                tf.write(f"F1 Score: {f1:.4f}\n")
                tf.write(f"Recall: {recall:.4f}\n")
                tf.write(f"Accuracy: {val_accuracy:.2f}%\n")
                tf.write("Confusion Matrix:\n")
                max_val_len = max(len(str(val)) for row_ in cm for val in row_)
                for row in cm:
                    formatted_row = "[" + ", ".join(f"{val:>{max_val_len}d}" for val in row) + "]"
                    tf.write(formatted_row + "\n")
    return metrics


def validate_model(
    model: torch.nn.Module, 
    val_loader: torch.utils.data.DataLoader, 
    criterion: Union[torch.nn.Module, str],
    checkpoint_path: Optional[str] = None, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    device: str = 'cpu', 
    metrics_flag: bool = False,
    use_one_hot: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """
    Validate a model on a given validation dataset.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (Union[torch.nn.Module, str]): The loss function to evaluate the model. Can be a string or torch.nn.Module.
        checkpoint_path (Optional[str]): Path to a checkpoint file to load the model state. Defaults to None.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load the state into if checkpoint_path is provided. Defaults to None.
        device (str): The device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        metrics_flag (bool): Flag to compute and save additional metrics during validation. Defaults to False.
        use_one_hot (bool): Whether to use one-hot encoding for targets. Defaults to False.

    Returns:
        Tuple[Optional[float], Optional[float]]: Tuple containing:
            - val_loss: Validation loss value, or None if loading checkpoint failed.
            - val_accuracy: Validation accuracy percentage, or None if loading checkpoint failed.
    
    Notes:
        - If checkpoint_path is provided, the model (and optimizer if provided) will be loaded from that checkpoint.
        - If optimizer is provided with checkpoint_path, the function will attempt to map the optimizer 
          class to a known optimizer type from the mapper module.
        - For string criterion, it strips any parameters in parentheses and uses the mapper to get the criterion.
    """
    if checkpoint_path:
        try:
            # If optimizer is provided, extract its type and parameters
            optimizer_type = None
            optimizer_params = {}
            if optimizer is not None:
                # Map the optimizer class to its name in the OPTIMIZERS dictionary
                for name, opt_class in mapper.OPTIMIZERS.items():
                    if isinstance(optimizer, opt_class):
                        optimizer_type = name
                        break
                
                if optimizer_type is None:
                    # Fallback to class name if not found in mapper
                    optimizer_type = optimizer.__class__.__name__
                    print(f"Warning: Optimizer class {optimizer_type} not found in mapper, using class name directly.")
                
                # Get optimizer parameters from its defaults
                optimizer_params = optimizer.defaults
                
            model, optimizer, _, _, _ = checkpoint.load_model_checkpoint(
                model, 
                device, 
                checkpoint_path, 
                optimizer_type, 
                optimizer_params
            )
            print(f"Model loaded from checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(e)
            return None, None
    else:
        model.to(device)
        model.eval()

    if isinstance(criterion, str):
        criterion_name = re.match(r'^([^(]+)', criterion).group(1).strip()
        criterion_name = criterion_name.split('.')[-1] 
        criterion = mapper.get_criterion(criterion_name, {})

    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating", unit="batch"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate loss for this batch
            if use_one_hot:
                targets = torch.nn.functional.one_hot(target, output.size(-1)).to(output.dtype)
                batch_loss = criterion(output, targets).item()
            else:
                batch_loss = criterion(output, target).item()
                
            val_loss += batch_loss * data.size(0)  # Weighted by batch size
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += data.size(0)
            
            # Store predictions and targets for metrics calculation
            if metrics_flag:
                all_preds.extend(predicted.cpu())
                all_targets.extend(target.cpu())
    
    # Calculate average loss and accuracy
    val_loss = val_loss / total_samples
    val_accuracy = 100 * correct_predictions / total_samples

    # Compute additional metrics if requested
    if metrics_flag and all_preds and all_targets:
        all_preds = torch.tensor(all_preds)
        all_targets = torch.tensor(all_targets)
        compute_and_serialize_metrics(all_targets, all_preds, val_accuracy, checkpoint_path=checkpoint_path, save_txt=SAVE_METRICS_IN_TXT)

    return val_loss, val_accuracy


def train_and_validate_model(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    criterion: Union[str, torch.nn.Module], 
    optimizer: Union[str, torch.optim.Optimizer], 
    scheduler: Union[str, torch.optim.lr_scheduler._LRScheduler, None],
    device: str, 
    checkpoint_dir: str, 
    epochs: int = 30, 
    start_epoch: int = 0, 
    patience: Optional[int] = None,
    learning_rate: float = 0.001,
    early_stopping: bool = False,
    scheduler_params: Optional[Dict[str, Any]] = None,
    save_every: Optional[int] = None,
    use_one_hot: bool = False
) -> Dict[str, List[float]]:

    """
    Train and validate a PyTorch model for a specified number of epochs, saving checkpoints and logs.

    Args:
        model (torch.nn.Module): The model to train and validate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (Union[str, torch.nn.Module]): The loss function for training. Can be a string or an instance of nn.Module.
        optimizer (Union[str, torch.optim.Optimizer]): The optimizer for training. Can be a string or an instance of Optimizer.
        scheduler (Union[str, torch.optim.lr_scheduler._LRScheduler, None]): The scheduler for training. 
            Can be a string, an instance of LRScheduler, or None.
        device (str): Device to use for training ('cpu' or 'cuda').
        checkpoint_dir (str): Directory to save checkpoints and logs.
        epochs (int): Total number of epochs to train. Defaults to 30.
        start_epoch (int): Starting epoch (useful for resuming training). Defaults to 0.
        patience (Optional[int]): Number of epochs with no improvement after which training will be stopped.
            Defaults to None (no early stopping).
        learning_rate (float): Learning rate for the optimizer when created from string. Defaults to 0.001.
        early_stopping (bool): Flag to enable early stopping. Defaults to False.
        scheduler_params (Optional[Dict[str, Any]]): Parameters for the scheduler when created from string.
        save_every (Optional[int]): If provided, save model checkpoint every this many epochs. Defaults to None.
        use_one_hot (bool): Whether to use one-hot encoding for targets in loss calculation. Defaults to False.

    Returns:
        Dict[str, List[float]]: Dictionary containing training and validation metrics per epoch.
    
    Notes:
        - If criterion, optimizer, or scheduler are provided as strings, they will be converted to 
          their respective objects using the mapper module.
        - The function creates 'best' and 'last' checkpoint directories and saves model states after each epoch.
        - When early stopping is enabled (patience is not None), training will stop if validation loss
          does not improve for the specified number of epochs.
        - Training and validation metrics are logged to JSON files for later visualization.
    """
    # Convert criterion from string to nn.Module if needed
    if isinstance(criterion, str):
        criterion_name = re.match(r'^([^(]+)', criterion).group(1).strip()
        criterion_name = criterion_name.split('.')[-1]  
        criterion = mapper.get_criterion(criterion_name, {})

    # Convert optimizer from string to torch.optim.Optimizer if needed
    if isinstance(optimizer, str):
        optimizer_name = re.match(r'^([^(]+)', optimizer).group(1).strip()
        optimizer_name = optimizer_name.split('.')[-1]
        optimizer_args = {"lr": learning_rate}
        optimizer = mapper.get_optimizer(optimizer_name, model.parameters(), optimizer_args)

    # Convert scheduler from string to torch.optim.lr_scheduler.LRScheduler if needed
    if isinstance(scheduler, str):
        scheduler_name = re.match(r'^([^(]+)', scheduler).group(1).strip()
        scheduler_name = scheduler_name.split('.')[-1]
        scheduler_args = scheduler_params or {}
        scheduler = mapper.get_scheduler(scheduler_name, optimizer, scheduler_args)

    # Create directory for current epoch's checkpoint using checkpoint API
    last_dir = os.path.join(checkpoint_dir, 'last')
    best_dir = os.path.join(checkpoint_dir, 'best')
    os.makedirs(last_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
    # Create periodic checkpoint directory if save_every is specified
    if save_every is not None:
        periodic_dir = os.path.join(checkpoint_dir, 'periodic')
        os.makedirs(periodic_dir, exist_ok=True)
    
    best_loss = float('inf')
    count_epochs = 0
    
    for epoch in range(start_epoch, epochs):

        model.to(device)
        model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Training loop
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]") as train_bar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                output = model(data)
                if use_one_hot:
                    targets = torch.nn.functional.one_hot(target, output.size(-1)).to(output.dtype)
                    loss = criterion(output, targets)
                else:
                    loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy
                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct_predictions += (predicted == target).sum().item()
                total_samples += data.size(0)

                # Update progress bar
                train_bar.update(1)
                train_bar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': 100. * correct_predictions / total_samples
                })

        # Compute epoch statistics
        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validate with metrics if in the last epoch or at specific intervals
        metrics_flag = (epoch == epochs - 1)  # Enable metrics in last epoch
        val_loss, val_accuracy = validate_model(
            model=model, 
            val_loader=val_loader, 
            criterion=criterion, 
            checkpoint_path=None, 
            device=device,
            metrics_flag=metrics_flag,
            use_one_hot=use_one_hot
        )
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save last checkpoint
        checkpoint.save_model_checkpoint(epoch_dir=last_dir, model=model, optimizer=optimizer, epoch=epoch, loss=epoch_loss)
        
        # Save periodic checkpoint if requested
        if save_every is not None and (epoch + 1) % save_every == 0:
            periodic_epoch_dir = os.path.join(periodic_dir, f'epoch_{epoch + 1}')
            os.makedirs(periodic_epoch_dir, exist_ok=True)
            checkpoint.save_model_checkpoint(epoch_dir=periodic_epoch_dir, model=model, optimizer=optimizer, epoch=epoch, loss=epoch_loss)
        
        # Handle best model saving and early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint.save_model_checkpoint(epoch_dir=best_dir, model=model, optimizer=optimizer, epoch=epoch, loss=val_loss)
            count_epochs = 0
        else:
            count_epochs += 1
            
        # Apply early stopping if enabled and patience is exceeded
        if (early_stopping or patience is not None) and count_epochs >= (patience or 5):
            print(f"Early stopping at epoch {epoch + 1}. No improvement in validation loss for {count_epochs} epochs.")
            break
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log metrics
        update_logs(log_file=os.path.join(checkpoint_dir, "loss_logs.json"), epoch=epoch + 1, training_metric=epoch_loss, validation_metric=val_loss, metric_name="loss")
        update_logs(log_file=os.path.join(checkpoint_dir, "accuracy_logs.json"), epoch=epoch + 1, training_metric=epoch_accuracy, validation_metric=val_accuracy, metric_name="accuracy")
    
    # Return collected metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # Populate metrics from log files if they exist
    loss_log_path = os.path.join(checkpoint_dir, "loss_logs.json")
    acc_log_path = os.path.join(checkpoint_dir, "accuracy_logs.json")
    
    if os.path.exists(loss_log_path):
        with open(loss_log_path, "r") as f:
            loss_logs = json.load(f)
            for entry in loss_logs:
                if "training_loss" in entry:
                    metrics["train_loss"].append(entry["training_loss"])
                if "validation_loss" in entry:
                    metrics["val_loss"].append(entry["validation_loss"])
    
    if os.path.exists(acc_log_path):
        with open(acc_log_path, "r") as f:
            acc_logs = json.load(f)
            for entry in acc_logs:
                if "training_accuracy" in entry:
                    metrics["train_accuracy"].append(entry["training_accuracy"])
                if "validation_accuracy" in entry:
                    metrics["val_accuracy"].append(entry["validation_accuracy"])
    
    return metrics


