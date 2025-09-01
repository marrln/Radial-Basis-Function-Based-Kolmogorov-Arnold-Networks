import io
import os
from thop import profile  
import sys
import pandas as pd
import ast
import torch
from torchinfo import summary
from checkpoint_config import extract_hyperparameters_from_directories, checkpoint_dir_name 

def count_parameters(model):
    """
    Counts the total and trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.s

    Returns:
        tuple: A tuple containing:
            - total_params (int): The total number of parameters in the model.
            - trainable_params (int): The number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_attributes(model, root_dir, dimension_list: list, grid_size: int, lr: float, sched : str, optim: str, 
                    criterion: str, grid_min: float, grid_max: float, inv_denominator: float, 
                    x_dim: int, y_dim: int, channel_size: int, seed: int):
    """
    Saves the attributes and details of a PyTorch model to a text file, including
    its parameter counts, architecture summary, and MACs (Multiply-Accumulate Operations).

    Args:
        model (torch.nn.Module): The PyTorch model.
        root_dir (str): The root directory where model checkpoint folders are located.
        dimension_list (list): The list of dimensions representing the model architecture.
        grid_size (int): The size of the grid used for training or evaluation.
        lr (float): The learning rate used for training the model.
        sched (str): Scheduler to use (e.g. 'ReduceOnPlateau', 'ExponentialLR').
        optim (str): The optimizer name (e.g., 'Adam', 'SGD').
        criterion (str): The loss function used (e.g., 'CrossEntropyLoss').
        grid_min (float): The minimum grid value.
        grid_max (float): The maximum grid value.
        inv_denominator (float): The denominator used for inverse scaling.
        x_dim (int): The width of the input image.
        y_dim (int): The height of the input image.
        channel_size (int): The number of channels in the input image.
        seed (int, optional): The random seed used for reproducibility.

    Returns:
        str: The directory path where the attributes were saved.
    """
    # Count total and trainable parameters in the model
    total_params, trainable_params = count_parameters(model)

    # Redirect stdout to capture model summary
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Generate the model summary using torchinfo
    input_size = (1, channel_size*x_dim*y_dim)  # Example input size
    if dimension_list[0] != channel_size*x_dim*y_dim:
        raise ValueError("The first dimension of the dimension list must match the input size. Got {} but expected {}.".format(dimension_list[0], channel_size*x_dim*y_dim))
    summary(model, input_size=input_size)
    model_summary = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Generate a summary again for direct usage (for compatibility)
    model_summary = summary(model, input_size=input_size)

    # Calculate the number of MACs
    input_tensor = torch.randn(*input_size)  # Example input tensor
    macs, params = profile(model.cpu(), inputs=(input_tensor,))

    # Construct the directory path for saving the model attributes
    dir_path = checkpoint_dir_name(
        criterion=criterion, optimizer=optim, scheduler=sched, seed=seed, learning_rate=lr,
        grid_size=grid_size, grid_min=grid_min, grid_max=grid_max,
        inv_denominator=inv_denominator, dim_list=dimension_list, root_dir=root_dir
    )
    os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist

    # File path for the attributes text file
    file_path = os.path.join(dir_path, "model_attributes.txt")

    # Save the model attributes to the file
    with open(file_path, "w", encoding='utf-8') as file:
        file.write("Model Attributes:\n")
        file.write(f"Dimension List: {dimension_list}\n")
        file.write(f"Number of FasterKAN Layers: {len(dimension_list) - 1}\n")
        file.write(f"Grid Size: {grid_size}\n")
        file.write(f"Grid Min: {grid_min}\n")
        file.write(f"Grid Max: {grid_max}\n")
        file.write(f"Inv Denominator: {inv_denominator}\n")
        file.write(f"Total Parameters: {total_params}\n")
        file.write(f"Trainable Parameters: {trainable_params}\n")
        file.write(f"MACs (Multiply-Accumulate Operations): {macs}\n")
        file.write("\nModel Summary:\n")
        file.write(str(model_summary))
    
    return dir_path

def process_model_data(root_dir):
    """
    Processes model data from directories, extracts relevant information, and saves it to an Excel file.

    Args:
        root_dir (str): The root directory where model checkpoint folders are located.
    
    The function performs the following steps:
    1. Extracts hyperparameter values from the directory structure using the `extract_hyperparameters_from_directories` function.
    2. Loops through all possible combinations of hyperparameters (learning rate, grid values, loss function, etc.).
    3. Constructs the full directory path for each combination using `checkpoint_dir_name`.
    4. Ensures that the relevant files (`model_attributes.txt`, `accuracy_logs.txt`, `loss_logs.txt`) exist in the checkpoint directory.
    5. Parses the content of these files to extract relevant metrics (total parameters, training/validation accuracies, losses, etc.).
    6. Appends the data for each epoch into a list, which is then converted into a Pandas DataFrame.
    7. Saves the DataFrame to an Excel file (`model_summary_final.xlsx`).

    Returns:
        None: The data is saved as an Excel file.
    """

    # Get all hyperparameter values from directories using the previously defined function
    hyperparams = extract_hyperparameters_from_directories(root_dir)

    data = []  # Initialize an empty list to store the processed data

    # Loop through all hyperparameter combinations
    for grid_min, grid_max in hyperparams["grid_min_max_list"]:
        for dimension_list in hyperparams["dim_lists"]:
            for lr in hyperparams["learning_rates"]:
                for criterion in hyperparams["loss_criterions"]:
                    for inv_denominator in hyperparams["inv_denominator_values"]:
                        for seed in hyperparams["seeds"]:
                            for optimizer in hyperparams["optimizers"]:
                                # Compute number of layers (excluding input/output dimensions)
                                num_layers = len(ast.literal_eval(dimension_list)) - 1
                                num_hidden = ast.literal_eval(dimension_list)[1]  # Assuming the first hidden dimension

                                # Use checkpoint_dir_name to construct the path
                                model_dir = checkpoint_dir_name(
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    seed=seed,
                                    dim_list=ast.literal_eval(dimension_list),
                                    learning_rate=lr,
                                    grid_size=hyperparams["grid_sizes"][0],  # Assuming single grid size for now
                                    grid_min=grid_min,
                                    grid_max=grid_max,
                                    inv_denominator=inv_denominator,
                                    root_dir=root_dir
                                )

                                # Paths to the specific files
                                attributes_file = os.path.join(model_dir, "model_attributes.txt")
                                accuracy_file = os.path.join(model_dir, "accuracy_logs.txt")
                                loss_file = os.path.join(model_dir, "loss_logs.txt")

                                # Ensure all files exist
                                if not (os.path.isfile(attributes_file) and os.path.isfile(accuracy_file) and os.path.isfile(loss_file)):
                                    print(f"Files missing in {model_dir}. Skipping.")
                                    continue

                                # Parse "model_attributes.txt" to extract relevant attributes
                                with open(attributes_file, 'r', encoding='utf-8') as f:
                                    attributes = f.read()

                                total_params = int(attributes.split("Total Parameters: ")[1].split("\n")[0])
                                trainable_params = int(attributes.split("Trainable Parameters: ")[1].split("\n")[0])
                                macs = float(attributes.split("MACs (Multiply-Accumulate Operations): ")[1].split("\n")[0])

                                # Parse "accuracy_logs.txt" to extract training and validation accuracies
                                with open(accuracy_file, 'r') as f:
                                    accuracy_data = f.readlines()

                                training_accuracies = [float(line.split("Training Accuracy = ")[1].split(",")[0]) for line in accuracy_data]
                                validation_accuracies = [float(line.split("Validation Accuracy = ")[1]) for line in accuracy_data]

                                # Parse "loss_logs.txt" to extract training and validation losses
                                with open(loss_file, 'r') as f:
                                    loss_data = f.readlines()

                                training_losses = [float(line.split("Training Loss = ")[1].split(",")[0]) for line in loss_data]
                                validation_losses = [float(line.split("Validation Loss = ")[1]) for line in loss_data]

                                # Format grid_min_max string
                                grid_min_max = f"({grid_min},{grid_max})"

                                # Append the data for each epoch
                                for epoch, (train_acc, val_acc, train_loss, val_loss) in enumerate(zip(training_accuracies, validation_accuracies, training_losses, validation_losses), start=1):
                                    data.append({
                                        "Num Layers": num_layers,
                                        "Hidden Dim": num_hidden,
                                        "Grid Size": hyperparams["grid_sizes"][0],  # NOTE: Assuming grid_size is constant across all paths
                                        "Epoch": epoch,
                                        "Learning Rate": lr,
                                        "Grid Min,Max": grid_min_max,
                                        "Inv Denominator": inv_denominator,
                                        "MACs": macs,
                                        "Total Parameters": total_params,
                                        "Trainable Parameters": trainable_params,
                                        "Training Accuracy": train_acc,
                                        "Validation Accuracy": val_acc,
                                        "Training Loss": train_loss,
                                        "Validation Loss": val_loss
                                    })

    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    output_file = "model_summary_final.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Data has been saved to {output_file}")