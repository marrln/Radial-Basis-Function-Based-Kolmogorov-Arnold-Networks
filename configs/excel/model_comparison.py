import os
import pandas as pd
import ast
from configs.checkpoint_handlers.checkpoint_config import extract_hyperparameters_from_directories, checkpoint_dir_name

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
                                        "Grid Size": hyperparams["grid_sizes"][0],  # Assuming grid_size is constant across all paths
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

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    output_file = "model_summary_final.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Data has been saved to {output_file}")
