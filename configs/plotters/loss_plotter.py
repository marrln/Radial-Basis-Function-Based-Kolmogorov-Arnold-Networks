import matplotlib.pyplot as plt
import os
import re
from configs.checkpoint_handlers.checkpoint_config import deconstruct_dir

def loss_plotter(dir_path):
    # Deconstruct the directory path to retrieve hyperparameters
    seed, criter, optimi, learning_rate, dim_list, grid_size, grid_min, grid_max, inv_denominator = deconstruct_dir(dir_path)
    
    file_path = os.path.join(dir_path, 'loss_logs.txt')
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Extract the data using regular expressions
    epochs = []
    training_acc = []
    validation_acc = []

    for line in data:
        match = re.match(r"Epoch (\d+): Training Loss = ([\d.]+), Validation Loss = ([\d.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            training_acc.append(float(match.group(2)))
            validation_acc.append(float(match.group(3)))
    
    # Find the minimum validation loss for this model
    min_validation_loss = min(validation_acc) if validation_acc else float('inf')

    # Find the epoch(s) where the validation loss is minimum
    min_loss_epoch = validation_acc.index(min_validation_loss)
    min_loss_value = validation_acc[min_loss_epoch]

    # Plot the data for the first num epochs
    num = 30
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:num], training_acc[:num], label='Training Loss', marker='o', color = 'deepskyblue')
    plt.plot(epochs[:num], validation_acc[:num], label='Validation Loss', marker='o', color = 'red')

    # Plot the minima with markers
    plt.scatter(epochs[min_loss_epoch], min_loss_value, color='black', edgecolor='white', s=100, zorder=5, label='Min Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Diagram')
    plt.figtext(0.5, 0.001, 
                f'Model Hyperparams:\nSeed = {seed}, Criterion = {criter}, Optimizer = {optimi},\n'
                f'Learning Rate = {learning_rate}, Grid Size = {grid_size}, # of FasterKAN Layers = {len(dim_list)-1},\n'
                f'Dimension List = {str(dim_list)}, Grid Min = {grid_min}, Grid Max = {grid_max}, Inv Denominator = {inv_denominator}', 
                ha='center', va='top', fontsize=10, wrap=True)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, f'loss_plot.jpg'))
    plt.show()