import os
import ast
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# def checkpoint_dir_name(criterion: str, 
#                         optimizer: str, 
#                         seed: int, 
#                         dim_list: list, # eg. [input_dim, 128, 128, output_dim]
#                         learning_rate: float, 
#                         grid_size: int, 
#                         grid_min: float, 
#                         grid_max: float, 
#                         inv_denominator: float,
#                         root_dir: str
#                         ):
    
#     hyperparameters = {
#         "criterion": criterion,
#         "optimizer": optimizer,
#         "seed": seed,
#         "learning_rate": learning_rate,
#         "grid_size": grid_size,
#         "grid_min": grid_min,
#         "grid_max": grid_max,
#         "inv_denominator": inv_denominator,
#         "dim_list": dim_list
#     }

#     checkpoint_dir = os.path.join(root_dir,
#                                 str(hyperparameters["seed"]),
#                                 str(hyperparameters["criterion"]),
#                                 str(hyperparameters["optimizer"]),
#                                 str(hyperparameters["learning_rate"]),
#                                 str(hyperparameters["dim_list"]),
#                                 str(hyperparameters["grid_size"]),
#                                 str(hyperparameters["grid_min"]),
#                                 str(hyperparameters["grid_max"]),
#                                 str(hyperparameters["inv_denominator"]))
#     return checkpoint_dir

def checkpoint_dir_name(criterion: str, 
                        optimizer: str, 
                        seed: int, 
                        dim_list: list,  # e.g., [input_dim, 128, 128, output_dim]
                        learning_rate: float, 
                        grid_size: int, 
                        grid_min: float, 
                        grid_max: float, 
                        inv_denominator: float,
                        root_dir: str):
    """
    Constructs a directory path for storing model checkpoints based on hyperparameters.

    Args:
        criterion (str): The loss function used.
        optimizer (str): The optimizer used.
        seed (int): The random seed.
        dim_list (list): The list of dimensions for the model layers.
        learning_rate (float): The learning rate.
        grid_size (int): The size of the grid.
        grid_min (float): The minimum grid value.
        grid_max (float): The maximum grid value.
        inv_denominator (float): The denominator for inverse scaling.
        root_dir (str): The root directory where checkpoints will be saved.

    Returns:
        str: The constructed directory path for the checkpoint.
    """
    try:
        # Ensure `dim_list` is properly formatted as a string
        dim_list_str = str(dim_list).replace(' ', '')  # Remove spaces for consistency
        
        checkpoint_dir = os.path.join(
            root_dir,
            str(seed),
            criterion,
            optimizer,
            f"{learning_rate:.1e}",   
            dim_list_str,
            str(grid_size),
            f"{grid_min:.1e}",        
            f"{grid_max:.1e}",        
            f"{inv_denominator:.1e}"  
        )
        return checkpoint_dir
    except Exception as e:
        raise ValueError(f"Error constructing checkpoint directory with provided parameters: {locals()}") from e

def deconstruct_dir(dir_path: str):
    """
    Deconstructs a directory path into its constituent hyperparameters.

    Args:
        dir_path (str): The directory path to deconstruct.

    Returns:
        tuple: A tuple containing:
            - seed (int): The random seed used.
            - criterion (str): The criterion (loss function).
            - optimizer (str): The optimizer used.
            - learning_rate (float): The learning rate.
            - dim_list (list): The dimensions of the model layers.
            - grid_size (int): The grid size.
            - grid_min (float): The minimum grid value.
            - grid_max (float): The maximum grid value.
            - inv_denominator (float): The denominator for inverse scaling.

    Raises:
        ValueError: If the directory path format is invalid or cannot be parsed.
    """
    try:
        parts = dir_path.split(os.sep)
        if len(parts) < 9:
            raise ValueError("Directory path must contain at least 9 parts for proper deconstruction.")

        seed = int(parts[-9])
        criterion = str(parts[-8])
        optimizer = str(parts[-7])
        learning_rate = float(parts[-6])
        dim_list = ast.literal_eval(parts[-5])
        grid_size = int(parts[-4])
        grid_min = float(parts[-3])
        grid_max = float(parts[-2])
        inv_denominator = float(parts[-1])

        return (seed, criterion, optimizer, learning_rate, dim_list, grid_size, grid_min, grid_max, inv_denominator)

    except (IndexError, ValueError, SyntaxError) as e:
        raise ValueError(f"Error deconstructing directory path: {dir_path}. Ensure it follows the expected format.") from e


def create_checkpoint_directory(epoch, checkpoint_dir):
    """
    Create a directory for the current epoch's checkpoint.

    Args:
        epoch (int): The current epoch number.
        checkpoint_dir (str): Base directory for saving checkpoints.

    Returns:
        str: The path to the created directory for the current epoch.
    """
    epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Optionally provide feedback about directory creation
    # if not os.path.exists(epoch_dir):
    #     print(f"Created new directory: {epoch_dir}")
    # else:
    #     print(f"Directory already exists: {epoch_dir}")
    
    return epoch_dir

def save_checkpoint(epoch_dir, model, optimizer, epoch, loss, best_val_loss=None):
    """
    Save the model and optimizer states, along with training metadata, to a checkpoint file.

    Args:
        epoch_dir (str): Directory where the checkpoint will be saved.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        loss (float): The training loss for the current epoch.
        best_val_loss (float, optional): The best validation loss seen so far. Defaults to None.
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
    
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer=None, checkpoint_path='invalid_path', device=device):
    """
    Load the checkpoint for a model and optionally an optimizer.
    
    Args:
        model (torch.nn.Module): The model to load the state_dict into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state_dict into.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to map the model and optimizer state_dict to.
        
    Returns:
        tuple: Contains:
            - model (torch.nn.Module): Model with loaded state_dict.
            - optimizer (torch.optim.Optimizer, optional): Optimizer with loaded state_dict if provided.
            - int: Start epoch (next epoch to resume training).
            - float: Loss from the checkpoint.
            - float: Best validation loss (if present in checkpoint), else None.
    """
    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        loss = checkpoint.get('loss', float('inf'))
        best_val_loss = checkpoint.get('best_val_loss', None)  # Default to None if not found
        
        print(f"Checkpoint loaded from {checkpoint_path}. Next epoch is {start_epoch}.")

        return model, optimizer, start_epoch, loss, best_val_loss
    else:
        raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint_path}")

def extract_hyperparameters_from_directories(root_dir):
    """
    Extracts all hyperparameter values from directory paths and returns a dictionary
    with the hyperparameters as keys and their corresponding unique values as lists.

    Args:
        root_dir (str): The root directory to search for the model directories.
    
    Returns:
        dict: Dictionary with hyperparameters as keys and lists of their unique values.
    """
    # Initialize dictionaries to store unique values for each hyperparameter
    hyperparams = {
        "grid_min_max_list": set(),
        "learning_rates": set(),
        "loss_criterions": set(),
        "inv_denominator_values": set(),
        "seeds": set(),
        "optimizers": set(),
        "dim_lists": set(),
        "grid_sizes": set()
    }

    # Traverse through all the directories in root_dir
    for root, dirs, _ in os.walk(root_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)

            try:
                # Extract hyperparameters using deconstruct_dir
                seed, criterion, optimizer, learning_rate, dim_list, grid_size, grid_min, grid_max, inv_denominator = deconstruct_dir(dir_path)
                
                # Add hyperparameters to the respective sets
                hyperparams["grid_min_max_list"].add((grid_min, grid_max))
                hyperparams["learning_rates"].add(learning_rate)
                hyperparams["loss_criterions"].add(criterion)
                hyperparams["inv_denominator_values"].add(inv_denominator)
                hyperparams["seeds"].add(seed)
                hyperparams["optimizers"].add(optimizer)
                hyperparams["dim_lists"].add(str(dim_list))  # Store as a string to avoid nested lists
                hyperparams["grid_sizes"].add(grid_size)

            except ValueError:
                # Handle the case when the directory path does not match the expected format
                print(f"Skipping directory due to format error: {dir_path}")
                continue

    # Convert sets to sorted lists
    for key in hyperparams:
        hyperparams[key] = sorted(list(hyperparams[key]))

    return hyperparams
