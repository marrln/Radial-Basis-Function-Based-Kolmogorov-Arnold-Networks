"""
This module provides functions to plot training/validation accuracy and loss curves
from JSON log files, with optional display of model hyperparameters.

Functions:
----------
- extract_log_data(file_path: str, keys: list) -> dict:
    Reads a JSON log file and extracts lists of values for specified keys across all epochs.
    Returns a dictionary mapping each key to its corresponding list of values, including the list of epochs.

- accuracy_plotter(dir_path, show_hyperparams=True):
    Generates and saves a plot of training, validation, and (optionally) test accuracy over epochs. 
    Highlights the epoch with the highest validation accuracy. 
    If only a single test accuracy is present, displays it as a horizontal line. 
    Optionally annotates the plot with model hyperparameters extracted from the directory name or configuration.

- loss_plotter(dir_path, show_hyperparams=True):
    Generates and saves a plot of training, validation, and (optionally) test loss over epochs. 
    Highlights the epoch with the lowest validation loss. If only a single test loss is present, displays it as a horizontal line. 
    Optionally annotates the plot with model hyperparameters.

- plot_confusion_matrix(y_test, y_pred, class_names, title=None, normalize=False, cmap=None):
    Plots a confusion matrix and a classification report as heatmaps.
    Supports normalization and custom colormap selection.
    Displays metrics per class and overall confusion matrix.

Notes:
------
1. Ensure your logs are saved as JSON files ('accuracy_logs.json' and 'loss_logs.json') in the target directory.
2. Call `accuracy_plotter(dir_path)` or `loss_plotter(dir_path)` with the directory containing the logs and config.
3. Plots are saved as 'accuracy_plot.jpg' and 'loss_plot.jpg' in the same directory and displayed interactively.
4. For confusion matrix plotting, provide true and predicted labels along with class names.

TODO: If the hyperparameters are too many, the figtext may overflow outside the figure. Implement a better way to handle this.
"""

from calendar import c
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Local imports
from . import checkpoint_utils as checkpoint

# GLOBAL CONSTANTS FOR AESTHETICS
MARKER_SIZE, MARKER_TYPE = 3, 'o'                                                                      # Marker size and type for accuracy/loss curves
TRAIN_COLOR, VALIDATION_COLOR, TEST_COLOR = 'deepskyblue', 'red', 'green'                              # Colors for training, validation, and test accuracy curves
MAX_MARKER_COLOR, MAX_MARKER_EDGECOLOR, MAX_MARKER_SIZE, MAX_MARKER_ZORDER = 'black', 'white', 100, 5  # Marker for max/min points
TEST_LINESTYLE, TEST_LINEWIDTH = '--', 1.5                                                             # Test line style (for single test value)
HLINE_COLOR, HLINE_LINESTYLE, HLINE_LABEL = 'gray', '--', '100% Accuracy'                              # 100% accuracy line
FIGSIZE, FIGTEXT_FONTSIZE = (8, 5), 10                                                                 # Figure properties
FIGTEXT_WRAP, FIGTEXT_X, FIGTEXT_Y, FIGTEXT_HA, FIGTEXT_VA = True, 0.5, 0.001, 'center', 'top'         # Horizontal alignment, vertical alignment for figtext


def extract_log_data(file_path: str, keys: list[str]) -> dict[str, list]:
    """
    Extracts lists of values for specified metric keys from a JSON log file.

    Args:
        file_path (str): Path to the JSON log file containing a list of epoch dictionaries.
        keys (list[str]): Metric keys to extract (e.g., ['training_accuracy', 'validation_accuracy']).

    Returns:
        dict[str, list]: Dictionary mapping each key to a list of values per epoch, plus 'epochs' for epoch numbers.
    """
    result = {key: [] for key in keys}
    result['epochs'] = []
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Log file does not contain a list of epoch dictionaries.")
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError("Each entry in the log file must be a dictionary.")
            result['epochs'].append(entry.get('epoch'))
            for key in keys:
                result[key].append(entry.get(key, None))
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file: {file_path}")
    except Exception as e:
        print(f"Error while extracting log data: {e}")
    return result


def accuracy_plotter(dir_path: str, config_path: str = None, show_hyperparams: bool = True) -> None:
    """
    Plots training and validation accuracy from a JSON log file.
    If test accuracy is present, plots it as well. If only one test accuracy is present (best model), shows it as a horizontal line.

    Args:
        dir_path (str): Path to the directory containing the JSON log files.
        config_path (str, optional): Path to the config file. If None, will look for config.json in dir_path.
        show_hyperparams (bool): If True, displays model hyperparameters as figtext on the plot.

    Saves:
        A plot 'accuracy_plot.jpg' in the same directory.
    """
    try:
        # If config_path is not provided, look for config.json in dir_path
        if config_path is None:
            config_path = os.path.join(dir_path, 'config.json')
        
        # Use read_config function from checkpoint_utils
        hyperparams = checkpoint.read_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        hyperparams = {}

    file_path = os.path.join(dir_path, 'accuracy_logs.json')
    log_data = extract_log_data(file_path, ['training_accuracy', 'validation_accuracy', 'test_accuracy'])

    epochs = log_data['epochs']
    training_acc = log_data['training_accuracy']
    validation_acc = log_data['validation_accuracy']
    test_acc = log_data['test_accuracy']

    # Filter out None values for realistic plotting
    valid_val_acc = [(i, v) for i, v in enumerate(validation_acc) if v is not None]
    if not epochs or not training_acc or not valid_val_acc:
        print("Error: Missing or empty accuracy log data.")
        return

    max_acc_idx, max_validation_acc = max(valid_val_acc, key=lambda x: x[1])
    max_acc_epoch = max_acc_idx
    max_acc_value = max_validation_acc

    plt.figure(figsize=FIGSIZE)
    plt.plot(epochs, training_acc, label='Training Accuracy', marker=MARKER_TYPE, color=TRAIN_COLOR, markersize=MARKER_SIZE)
    plt.plot(epochs, validation_acc, label='Validation Accuracy', marker=MARKER_TYPE, color=VALIDATION_COLOR, markersize=MARKER_SIZE)
    plt.scatter(epochs[max_acc_epoch], max_acc_value, color=MAX_MARKER_COLOR, edgecolor=MAX_MARKER_EDGECOLOR, s=MAX_MARKER_SIZE, zorder=MAX_MARKER_ZORDER, label='Max Validation Accuracy')

    if test_acc:
        if len(test_acc) == 1:
            plt.axhline(y=test_acc[0], color=TEST_COLOR, linestyle=TEST_LINESTYLE, linewidth=TEST_LINEWIDTH, label=f'Test Accuracy ({test_acc[0]:.2f}%)')
        elif len(test_acc) == len(epochs):
            plt.plot(epochs, test_acc, label='Test Accuracy', marker=MARKER_TYPE, color=TEST_COLOR, markersize=MARKER_SIZE)
        else:
            test_epochs = [epochs[i] for i, val in enumerate(test_acc) if val is not None]
            test_acc_filtered = [val for val in test_acc if val is not None]
            plt.plot(test_epochs, test_acc_filtered, label='Test Accuracy', marker=MARKER_TYPE, color=TEST_COLOR, markersize=MARKER_SIZE)

    plt.axhline(y=100, color=HLINE_COLOR, linestyle=HLINE_LINESTYLE, label=HLINE_LABEL)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Diagram')
    if show_hyperparams:
        hp_str = '\n'.join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.figtext(FIGTEXT_X, FIGTEXT_Y, f'Model Hyperparams:\n{hp_str}', ha=FIGTEXT_HA, va=FIGTEXT_VA, fontsize=FIGTEXT_FONTSIZE, wrap=FIGTEXT_WRAP)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, f'accuracy_plot.jpg'))
    plt.show()


def loss_plotter(dir_path: str, config_path: str = None, show_hyperparams: bool = True) -> None:
    """
    Plots training and validation loss from a JSON log file.
    If test loss is present, plots it as well. If only one test loss is present (best model), shows it as a horizontal line.

    Args:
        dir_path (str): Path to the directory containing the JSON log files.
        config_path (str, optional): Path to the config file. If None, will look for config.json in dir_path.
        show_hyperparams (bool): If True, displays model hyperparameters as figtext on the plot.

    Saves:
        A plot 'loss_plot.jpg' in the same directory.
    """
    try:
        # If config_path is not provided, look for config.json in dir_path
        if config_path is None:
            config_path = os.path.join(dir_path, 'config.json')
        
        # Use read_config function from checkpoint_utils
        hyperparams = checkpoint.read_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        hyperparams = {}

    file_path = os.path.join(dir_path, 'loss_logs.json')
    log_data = extract_log_data(file_path, ['training_loss', 'validation_loss', 'test_loss'])

    epochs = log_data['epochs']
    training_loss = log_data['training_loss']
    validation_loss = log_data['validation_loss']
    test_loss = log_data['test_loss']

    # Filter out None values for realistic plotting
    valid_val_loss = [(i, v) for i, v in enumerate(validation_loss) if v is not None]
    if not epochs or not training_loss or not valid_val_loss:
        print("Error: Missing or empty loss log data.")
        return

    min_loss_idx, min_validation_loss = min(valid_val_loss, key=lambda x: x[1])
    min_loss_epoch = min_loss_idx
    min_loss_value = min_validation_loss

    plt.figure(figsize=FIGSIZE)
    plt.plot(epochs, training_loss, label='Training Loss', marker=MARKER_TYPE, color=TRAIN_COLOR, markersize=MARKER_SIZE)
    plt.plot(epochs, validation_loss, label='Validation Loss', marker=MARKER_TYPE, color=VALIDATION_COLOR, markersize=MARKER_SIZE)
    plt.scatter(epochs[min_loss_epoch], min_loss_value, color=MAX_MARKER_COLOR, edgecolor=MAX_MARKER_EDGECOLOR, s=MAX_MARKER_SIZE, zorder=MAX_MARKER_ZORDER, label='Min Validation Loss')

    if test_loss:
        if len(test_loss) == 1:
            plt.axhline(y=test_loss[0], color=TEST_COLOR, linestyle=TEST_LINESTYLE, linewidth=TEST_LINEWIDTH, label=f'Test Loss ({test_loss[0]:.4f})')
        elif len(test_loss) == len(epochs):
            plt.plot(epochs, test_loss, label='Test Loss', marker=MARKER_TYPE, color=TEST_COLOR, markersize=MARKER_SIZE)
        else:
            test_epochs = [epochs[i] for i, val in enumerate(test_loss) if val is not None]
            test_loss_filtered = [val for val in test_loss if val is not None]
            plt.plot(test_epochs, test_loss_filtered, label='Test Loss', marker=MARKER_TYPE, color=TEST_COLOR, markersize=MARKER_SIZE)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Diagram')
    if show_hyperparams:
        hp_str = '\n'.join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.figtext(FIGTEXT_X, FIGTEXT_Y, f'Model Hyperparams:\n{hp_str}', ha=FIGTEXT_HA, va=FIGTEXT_VA, fontsize=FIGTEXT_FONTSIZE, wrap=FIGTEXT_WRAP)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, f'loss_plot.jpg'))
    plt.show()

def plot_confusion_matrix(y_test, y_pred, class_names, title=None, normalize=False, cmap=None, cbar=True):
    """
    Plots a confusion matrix and classification report.
    Args:
        y_test: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        title: Optional plot title.
        normalize: If True, normalize confusion matrix rows.
        cmap: Colormap for confusion matrix.
        cbar: If True, display color bar.
    """
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cmap = cmap if cmap is not None else 'Blues' 

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Replace NaN with 0 for rows with sum 0

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title if title is not None else "", fontsize=16, fontweight='bold')

    # Subplot 1: Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report)
    # Exclude 'accuracy' and 'macro avg' from the classification report DataFrame
    report_df = report_df.drop(columns=['accuracy'], errors='ignore')
    report_df = report_df.drop(columns=['macro avg'], axis=1, errors='ignore')

    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap=cmap, fmt=".2f", cbar=cbar, ax=ax1)
    ax1.set_title("Metrics per Class")

    # Subplot 2: Confusion matrix
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=cbar,
        ax=ax2
    )
    ax2.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    # Final layout
    plt.tight_layout() 
    plt.show()