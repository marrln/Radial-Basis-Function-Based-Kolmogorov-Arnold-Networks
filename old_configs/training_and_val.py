import torch
import os
import re
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from tqdm.notebook import tqdm 
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from checkpoint_config import create_checkpoint_directory, save_checkpoint, load_checkpoint # type: ignore
from fasterkan import FasterKAN # type: ignore
from model_comparison import save_attributes # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Mapping of schedulers names to their corresponding PyTorch classes
SCHEDULERS = {
    "StepLR": lr_scheduler.StepLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "LinearLR": lr_scheduler.LinearLR,
    # Add other optimizers here as needed
}

# Mapping of optimizer names to their corresponding PyTorch classes
OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "AdamW": optim.AdamW,
    # Add other optimizers here as needed
}

# Mapping of criterion names to their corresponding PyTorch classes
CRITERIONS = {
    'BCELoss' : nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "HuberLoss" : nn.HuberLoss
    # Add other criterions here as needed
}

def initialize_model(root_dir, dimension, grid_size, lr, sched, optim, criterion, grid_min, grid_max, 
                    inv_denominator, x_dim, y_dim, channel_size, seed, model_type = FasterKAN):
    """
    Initialize the FasterKAN model with specified parameters.

    Args:
        root_dir (str): The root directory where model checkpoint folders are located.
        dimension (list or tuple): List or tuple specifying the number of units in each hidden layer.
        grid_size (int): Size of the grid.
        lr (float): Learning rate.
        sched (str): Scheduler to use (e.g. 'ReduceOnPlateau', 'ExponentialLR').
        optim (str): Optimizer to use (e.g., 'SGD', 'Adam').
        criterion (str): Loss function to use (e.g., 'CrossEntropyLoss', 'MSELoss').
        grid_min (float): Minimum value for the grid.
        grid_max (float): Maximum value for the grid.
        inv_denominator (float): Inverse denominator for the model.
        x_dim (int): Width of the input data.
        y_dim (int): Height of the input data.
        channel_size (int): Number of channels in the input data.
        seed (int): The seed used for reproductibility.

    Returns:
        model (torch.nn.Module): Initialized FasterKAN model.
        file_path (str): Path to the saved model attributes.
    """
    sched_str = str(sched)
    optim_str = str(optim)
    criterion_str = str(criterion)

    model = model_type(
        layers_hidden=dimension,
        num_grids=grid_size,
        grid_min=grid_min,
        grid_max=grid_max,
        inv_denominator=inv_denominator
    ).cpu()

    # Save model attributes and return the directory path
    file_path = None
    if root_dir is not None:
        file_path = save_attributes(
            model,
            root_dir=root_dir,
            dimension_list=dimension,
            grid_size=grid_size,
            lr=lr,
            sched=sched_str,
            optim=optim_str,
            criterion=criterion_str,
            grid_min=grid_min,
            grid_max=grid_max,
            inv_denominator=inv_denominator,
            x_dim=x_dim,
            y_dim=y_dim,
            channel_size=channel_size,
            seed=seed
        )
    model.to(device)

    return model, file_path


def update_logs(log_file, epoch, training_metric, validation_metric, metric_name):
    """
    Update log files for training and validation metrics.

    Args:
        log_file (str): Path to the log file.
        epoch (int): Current epoch.
        training_metric (float): Metric value for training (e.g., loss or accuracy).
        validation_metric (float): Metric value for validation (e.g., loss or accuracy).
        metric_name (str): Name of the metric (e.g., "Loss", "Accuracy").
    """
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
        lines = [line for line in lines if not line.startswith(f"Epoch {epoch}:")]
    else:
        lines = []

    with open(log_file, "w") as f:
        f.writelines(lines)  # Retain existing lines
        f.write(f"Epoch {epoch}: Training {metric_name} = {training_metric:.4f}, Validation {metric_name} = {validation_metric:.4f}\n")

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
    # Load model and optimizer state if a checkpoint path is provided
    if checkpoint_path:
        try:
            model, optimizer, _, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device) 
            print(f"Model loaded from checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(e)
            return None, None
    else:
        model.to(device)
        model.eval()

    if isinstance(criterion, str):
        # Remove parentheses if provided, e.g., "CrossEntropyLoss()"
        criterion = re.sub(r"\(.*\)", "", criterion)
        if criterion not in CRITERIONS:
            raise ValueError(f"Criterion '{criterion}' is not recognized. Available options: {list(CRITERIONS.keys())}")
        criterion = CRITERIONS[criterion]()

    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating", unit="batch"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # loss = criterion(output, torch.nn.functional.one_hot(target,output.size(-1)).to(output.dtype))
            # val_loss += loss.item() * data.size(0)

            # _, predicted = torch.max(output, 1)
            # correct_predictions += (predicted == target).sum().item()
            # total_samples += data.size(0)

            # if metrics_flag:
            all_preds.extend(output.cpu())
            all_targets.extend(target.cpu())
            
        all_preds = torch.stack(all_preds).to(device)
        all_targets = torch.stack(all_targets).to(device)
        
        # print('Output example:', all_preds[0])
        # print('Tagret example:', all_targets[0], '->', torch.nn.functional.one_hot(all_targets[0],all_preds.size(-1)).to(all_preds.dtype))
        
        # val_loss = criterion(all_preds, all_targets)
        val_loss = criterion(all_preds, torch.nn.functional.one_hot(all_targets,all_preds.size(-1)).to(all_preds.dtype)).cpu().item()
        
        _, all_preds = torch.max(all_preds, dim=-1)
        val_accuracy = 100 * (all_preds == all_targets).sum().item() / len(all_targets)
        
    all_preds = all_preds.cpu()
    all_targets = all_targets.cpu()

    if metrics_flag:
        f1 = f1_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        cm = confusion_matrix(all_targets, all_preds)

        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {val_accuracy:.2f}%")
        print("Confusion Matrix:")
        print(cm)

        # Save metrics to a .txt file
        if checkpoint_path:
            metrics_path = os.path.join(os.path.dirname(checkpoint_path), "metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"Accuracy: {val_accuracy:.2f}%\n")
                f.write("Confusion Matrix:\n")
                for row in cm:
                    f.write(" ".join(map(str, row)) + "\n")

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
    debug_flag: bool = False
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
        debug_flag (bool, optional): Flag for debugging gradients. Defaults to False.
    """
    # Convert criterion from string to nn.Module if needed
    if isinstance(criterion, str):
        # Remove parentheses if provided, e.g., "CrossEntropyLoss()"
        criterion = re.sub(r"\(.*\)", "", criterion)
        if criterion not in CRITERIONS:
            raise ValueError(f"Criterion '{criterion}' is not recognized. Available options: {list(CRITERIONS.keys())}")
        criterion = CRITERIONS[criterion]()

    # Convert optimizer from string to torch.optim.Optimizer if needed
    if isinstance(optimizer, str):
        # Remove parentheses if provided
        optimizer = re.sub(r"\(.*\)", "", optimizer)
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Optimizer '{optimizer}' is not recognized. Available options: {list(OPTIMIZERS.keys())}")
        optimizer = OPTIMIZERS[optimizer](model.parameters())
        
    # Convert optimizer from string to torch.optim.Optimizer if needed
    if isinstance(scheduler, str):
        # Remove parentheses if provided
        scheduler = re.sub(r"\(.*\)", "", scheduler)
        if scheduler not in SCHEDULERS:
            raise ValueError(f"Optimizer '{scheduler}' is not recognized. Available options: {list(SCHEDULERS.keys())}")
        scheduler = SCHEDULERS[scheduler](optimizer)

    # Create directory for current epoch's checkpoint
    last_dir = create_checkpoint_directory('last', checkpoint_dir)
    best_dir = create_checkpoint_directory('best', checkpoint_dir)
    
    best_loss = float('inf')
    count_epochs = 0
    
    for epoch in range(start_epoch, epochs):
        # Set model to training mode
        model.to(device)
        model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Training loop
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]") as train_bar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(data)
                # loss = criterion(output, target)
                loss = criterion(output, torch.nn.functional.one_hot(target,output.size(-1)).to(output.dtype))

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                if debug_flag and batch_idx % 500 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")
                    for param in model.parameters():
                        if param.grad is not None:
                            print(f"Gradients of {param.shape}: {param.grad.mean()}")

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

        save_checkpoint(epoch_dir=last_dir, model=model, optimizer=optimizer, epoch=epoch, loss=epoch_loss)
        if (val_loss < best_loss):
            best_loss = val_loss
            save_checkpoint(epoch_dir=best_dir, model=model, optimizer=optimizer, epoch=epoch, loss=epoch_loss)
            count_epochs = 0
        elif patience is not None:
            count_epochs += 1
            
        if patience is not None and count_epochs >= patience:
            break
        
        if scheduler is not None:
            scheduler.step(val_loss)
            
        update_logs(log_file=os.path.join(checkpoint_dir, "loss_logs.txt"), epoch=epoch + 1, training_metric=epoch_loss, validation_metric=val_loss, metric_name="Loss")
        update_logs(log_file=os.path.join(checkpoint_dir, "accuracy_logs.txt"), epoch=epoch + 1, training_metric=epoch_accuracy, validation_metric=val_accuracy, metric_name="Accuracy")


