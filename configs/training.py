import torch
import os
import re
from tqdm.notebook import tqdm 
from sklearn.metrics import f1_score, confusion_matrix, recall_score
import mapper
import fasterkan
import checkpoint
import json


SAVE_METRICS_IN_TXT = True

def initialize_kan_model(
    root_dir: str,
    hyperparams: dict,
    x_dim: int,
    y_dim: int,
    channel_size: int,
    hyperparams_typedict: dict,
    device: str
) -> tuple[torch.nn.Module, str]:
    """
    Initialize the FasterKAN model with specified parameters.

    Args:
        root_dir (str): The root directory where model checkpoint folders are located.
        hyperparams (dict): Dictionary containing all model and training hyperparameters.
        x_dim (int): Width of the input data.
        y_dim (int): Height of the input data.
        channel_size (int): Number of channels in the input data.
        hyperparams_typedict (dict): Dictionary mapping hyperparameter names to types.
        device (str): Device to use for the model ('cpu' or 'cuda').

    Returns:
        model (torch.nn.Module): Initialized FasterKAN model.
        file_path (str): Path to the saved model attributes.
    """
    model = fasterkan.FasterKAN(
        layers_hidden=hyperparams['dim_list'],
        num_grids=hyperparams['grid_size_per_layer'],
        grid_min=hyperparams['grid_min'],
        grid_max=hyperparams['grid_max'],
        inv_denominator=hyperparams['inv_denominator']
    ).to(device)
    
    file_path = checkpoint.save_attributes(
        model,
        root_dir=root_dir,
        hyperparams=hyperparams,
        hyperparams_typedict=hyperparams_typedict,
        x_dim=x_dim,
        y_dim=y_dim,
        channel_size=channel_size
    )
    return model, file_path

def update_logs(
    log_file: str,
    epoch: int,
    training_metric: float,
    validation_metric: float,
    metric_name: str
) -> None:
    """
    Update log files for training and validation metrics in JSON format.

    Args:
        log_file (str): Path to the log file.
        epoch (int): Current epoch.
        training_metric (float): Metric value for training (e.g., loss or accuracy).
        validation_metric (float): Metric value for validation (e.g., loss or accuracy).
        metric_name (str): Name of the metric (e.g., "Loss", "Accuracy").
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
    checkpoint_path: str = None,
    save_txt: bool = SAVE_METRICS_IN_TXT
) -> dict:
    """
    Compute F1, recall, confusion matrix, and accuracy, print them, and save to JSON (and optionally TXT) files.

    Args:
        all_targets (Tensor or array-like): True labels.
        all_preds (Tensor or array-like): Predicted labels.
        val_accuracy (float): Validation accuracy (percentage).
        checkpoint_path (str, optional): Path to checkpoint for directory context. Defaults to None.
        save_txt (bool, optional): Whether to also save metrics as TXT. Defaults to True.

    Returns:
        dict: Dictionary containing computed metrics.
    """

    # NOTE: For multi-class classification, 'weighted' averages by support (number of true instances for each label),
    # while 'macro' gives equal weight to each class regardless of support.
    # Use 'weighted' if your classes are imbalanced and you want the metric to reflect the class distribution.
    # Use 'macro' if you want to treat all classes equally, regardless of their frequency.

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


def validate_model(model, val_loader, criterion, checkpoint_path=None, optimizer=None, device='cpu', metrics_flag=False):
    """
    Validate a model on a given validation dataset.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function to evaluate the model.
        checkpoint_path (str, optional): Path to a checkpoint file to load the model (and optionally optimizer) state. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into if checkpoint_path is provided. Defaults to None.
        device (str): The device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        metrics_flag (bool): Flag to compute additional metrics during validation. Defaults to False.

    Returns:
        tuple: Validation loss and validation accuracy.
    """
    if checkpoint_path:
        try:
            model, optimizer, _, _, _ = checkpoint.load_checkpoint(model, optimizer, checkpoint_path, device) 
            print(f"Model loaded from checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(e)
            return None, None
    else:
        model.to(device)
        model.eval()

    if isinstance(criterion, str):
        criterion = re.sub(r"\(.*\)", "", criterion)
        criterion = mapper.get_criterion(criterion, {})

    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating", unit="batch"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            all_preds.extend(output.cpu())
            all_targets.extend(target.cpu())
            
        all_preds = torch.stack(all_preds).to(device)
        all_targets = torch.stack(all_targets).to(device)
        
        val_loss = criterion(all_preds, torch.nn.functional.one_hot(all_targets,all_preds.size(-1)).to(all_preds.dtype)).cpu().item()
        
        _, all_preds = torch.max(all_preds, dim=-1)
        val_accuracy = 100 * (all_preds == all_targets).sum().item() / len(all_targets)
        
    all_preds = all_preds.cpu()
    all_targets = all_targets.cpu()

    if metrics_flag:
        compute_and_serialize_metrics(all_targets, all_preds, val_accuracy, checkpoint_path=checkpoint_path, save_txt=SAVE_METRICS_IN_TXT)

    return val_loss, val_accuracy


def train_and_validate_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler,
    device, 
    checkpoint_dir, 
    epochs: int = 30, 
    start_epoch: int = 0, 
    patience: int = None,
    # debug_flag: bool = False
):
    """
    Train and validate a PyTorch model for a specified number of epochs, saving checkpoints and logs.

    Args:
        model (torch.nn.Module): The model to train and validate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (str or torch.nn.Module): The loss function for training. Can be a string or an instance of nn.Module.
        optimizer (str or torch.optim.Optimizer): The optimizer for training. Can be a string or an instance of torch.optim.Optimizer.
        scheduler (str or torch.optim.lr_scheduler.LRScheduler) : The scheduler for training. Can be a string or an instance of torch.optim.lr_scheduler.LRScheduler.
        device (str): Device to use for training ('cpu' or 'cuda').
        checkpoint_dir (str): Directory to save checkpoints and logs.
        epochs (int, optional): Total number of epochs to train. Defaults to 30.
        start_epoch (int, optional): Starting epoch (useful for resuming training). Defaults to 0.
        # debug_flag (bool, optional): Flag for debugging gradients. Defaults to False.
    """
    # Convert criterion from string to nn.Module if needed
    if isinstance(criterion, str):
        criterion = re.sub(r"\(.*\)", "", criterion)
        criterion = mapper.get_criterion(criterion, {})

    # Convert optimizer from string to torch.optim.Optimizer if needed
    if isinstance(optimizer, str):
        optimizer = re.sub(r"\(.*\)", "", optimizer)
        optimizer = mapper.get_optimizer(optimizer, model.parameters(), {})

    # Convert scheduler from string to torch.optim.lr_scheduler.LRScheduler if needed
    if isinstance(scheduler, str):
        scheduler = re.sub(r"\(.*\)", "", scheduler)
        scheduler = mapper.get_scheduler(scheduler, optimizer, {})

    # Create directory for current epoch's checkpoint using checkpoint API
    last_dir = os.path.join(checkpoint_dir, 'last')
    best_dir = os.path.join(checkpoint_dir, 'best')
    os.makedirs(last_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
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
                loss = criterion(output, torch.nn.functional.one_hot(target,output.size(-1)).to(output.dtype))

                loss.backward()
                optimizer.step()

                # if debug_flag and batch_idx % 500 == 0:
                #     print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")
                #     for param in model.parameters():
                #         if param.grad is not None:
                #             print(f"Gradients of {param.shape}: {param.grad.mean()}")

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

        val_loss, val_accuracy = validate_model(model=model, val_loader=val_loader, criterion=criterion, checkpoint_path=None, device=device); 
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        checkpoint.save_model_checkpoint(epoch_dir=last_dir, model=model, optimizer=optimizer, epoch=epoch, loss=epoch_loss)
        if (val_loss < best_loss):
            best_loss = val_loss
            checkpoint.save_model_checkpoint(epoch_dir=best_dir, model=model, optimizer=optimizer, epoch=epoch, loss=epoch_loss)
            count_epochs = 0
        elif patience is not None:
            count_epochs += 1
            
        if patience is not None and count_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1} with patience {patience}. No improvement in validation loss for {count_epochs} epochs.")
            break
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        update_logs(log_file=os.path.join(checkpoint_dir, "loss_logs.json"), epoch=epoch + 1, training_metric=epoch_loss, validation_metric=val_loss, metric_name="loss")
        update_logs(log_file=os.path.join(checkpoint_dir, "accuracy_logs.json"), epoch=epoch + 1, training_metric=epoch_accuracy, validation_metric=val_accuracy, metric_name="accuracy")


