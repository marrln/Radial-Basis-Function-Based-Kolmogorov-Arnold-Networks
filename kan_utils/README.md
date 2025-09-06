# KAN Utilities

This folder contains the core implementation and utility modules for Radial Basis Function-Based Kolmogorov-Arnold Networks (RBF-KANs), specialized for image classification tasks.

## Core Components

- **fasterkan.py**  
  Implementation of the RBF-KAN architecture (FasterKAN) with optimizations for image classification tasks. Features include dropout with grid size-dependent rates, gradient boosting, and options for FPGA-friendly implementation.

- **training.py**  
  Comprehensive training pipeline for image classification, including model initialization from config files, metric calculation, validation routines, and checkpoint integration.

- **checkpoint_utils.py**  
  Utilities for managing model checkpoints, creating structured directories based on hyperparameters, saving/loading model states, and preserving configurations across training sessions.

- **experiment_eval.py**  
  Tools for evaluating image classification experiments, analyzing hyperparameter impact, and collecting performance metrics across multiple training runs.

- **mapper.py**  
  Maps string identifiers to PyTorch optimizers, schedulers, and loss functions for flexible configuration through JSON files.

- **plotter.py**  
  Visualization tools for training and validation metrics, with support for plotting accuracy and loss curves for image classification experiments.

- **general_utils.py**  
  Common utility functions used across the codebase, including configuration parsing, device management, and model analysis tools.

## Usage for Image Classification

The utilities in this folder are designed specifically for training and evaluating image classification models using the RBF-KAN architecture. The implementation has been tested on:

1. **MNIST** (handwritten digit classification)
2. **HAM10000** (skin lesion classification)

For examples of how to use these utilities in image classification tasks, see the notebooks in the `MNIST/` and `HAM10000/` directories.

## Quantization Integration

The training and checkpoint utilities in this folder integrate seamlessly with the quantization tools in the `quantization/` directory, enabling post-training quantization for deployment on resource-constrained devices.
