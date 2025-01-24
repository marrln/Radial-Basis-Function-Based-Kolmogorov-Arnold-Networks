import matplotlib.pyplot as plt
import os
import re
from configs.checkpoint_handlers.checkpoint_config import deconstruct_dir

def accuracy_plotter(dir_path):
    seed, criter, optimi, learning_rate, dim_list, grid_size, grid_min, grid_max, inv_denominator = deconstruct_dir(dir_path)
    file_path = os.path.join(dir_path, 'accuracy_logs.txt')
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Extract the data using regular expressions
    epochs = []
    training_acc = []
    validation_acc = []

    for line in data:
        match = re.match(r"Epoch (\d+): Training Accuracy = ([\d.]+), Validation Accuracy = ([\d.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            training_acc.append(float(match.group(2)))
            validation_acc.append(float(match.group(3)))
    
    # Find the maximum validation accuracy for this model
    max_validation_acc = max(validation_acc) if validation_acc else float('inf')

    # Find the epoch(s) where the validation accuracy is maximum
    max_acc_epoch = validation_acc.index(max_validation_acc)
    max_acc_value = validation_acc[max_acc_epoch]

    # Plot the data for the first num epochs
    num = 30
    plt.figure(figsize=(8, 5))

    plt.plot(epochs[:num], training_acc[:num], label='Training Accuracy', marker='o', color='deepskyblue')
    plt.plot(epochs[:num], validation_acc[:num], label='Validation Accuracy', marker='o', color='red')

    # Plot the minima with markers
    plt.scatter(epochs[max_acc_epoch], max_acc_value, color='black', edgecolor='white', s=100, zorder=5, label='Max Validation Accuracy')

    # Add a horizontal line at 100% accuracy
    plt.axhline(y=100, color='gray', linestyle='--', label='100% Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')  
    plt.title('Accuracy Diagram')

    plt.figtext(0.5, 0.001, 
                f'Model Hyperparams:\nSeed = {seed}, Criterion = {criter}, Optimizer = {optimi},\n'
                f'Learning Rate = {learning_rate}, Grid Size = {grid_size}, # of FasterKAN Layers = {len(dim_list)-1},\n'
                f'Dimension List = {str(dim_list)}, Grid Min = {grid_min}, Grid Max = {grid_max}, Inv Denominator = {inv_denominator}', 
                ha='center', va='top', fontsize=10, wrap=True)

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, f'accuracy_plot.jpg'))
    plt.show()