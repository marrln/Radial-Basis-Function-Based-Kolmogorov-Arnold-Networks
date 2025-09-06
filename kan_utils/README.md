# KAN Utilities for Image Classification

This folder contains the core implementation and utility modules for Radial Basis Function-Based Kolmogorov-Arnold Networks (RBF-KANs), specialized for image classification tasks.

## Core Components

- **fasterkan.py**  
  Implementation of the RBF-KAN architecture (FasterKAN) optimized for image classification. Features include adaptive dropout rates based on grid size, FPGA-friendly linear layers without bias terms, and modified backward pass with gradient boosting for grid and inverse denominator parameters.

- **training.py**  
  End-to-end training pipeline with model initialization, training loops, validation, early stopping, checkpoint integration, and comprehensive metrics computation for image classification tasks.

- **checkpoint_utils.py**  
  Robust checkpoint management system that handles model serialization, hyperparameter-based directory structures, config validation, and model state loading/saving with proper device mapping.

- **experiment_eval.py**  
  Tools for evaluating image classification experiments, analyzing hyperparameter impact, and collecting performance metrics across multiple training runs.

- **mapper.py**  
  Maps string identifiers to PyTorch optimizers, schedulers, and loss functions for flexible configuration through JSON files.

- **plotter.py**  
  Visualization tools for training and validation metrics, with support for plotting accuracy and loss curves for image classification experiments.

- **general_utils.py**  
  Common utility functions used across the codebase, including configuration parsing, device management, and model analysis tools.

## Usage for Image Classification

The utilities in this folder are specifically designed for training and evaluating image classification models using the RBF-KAN architecture. The workflow typically includes:

1. **Configuration**: Create a JSON config file with model hyperparameters
2. **Model Initialization**: Use `initialize_kan_model_from_config()` to create a model
3. **Training**: Run `train_and_validate_model()` with appropriate data loaders
4. **Evaluation**: Analyze results with `validate_model()` and visualization tools

The implementation has been optimized and tested on image classification datasets:

- **MNIST**: Handwritten digit classification (grayscale images)
- **HAM10000**: Skin lesion classification (dermatoscopic images)

## Integration with Quantization

The architecture supports post-training quantization through the tools in the `quantization/` directory. The checkpoint system is designed to preserve model states in a format compatible with the quantization workflow.

## Key Features for Image Classification

1. **Grid-Based RBF Implementation**: Uses radial basis functions with learned grid points and inverse denominators
2. **Adaptive Dropout**: Dropout rate that scales with grid size to prevent overfitting in larger models
3. **FPGA-Friendly Design**: Option to disable bias terms for hardware implementation
4. **Gradient Boosting**: Enhanced backward pass that prioritizes grid and inverse denominator parameter updates
5. **Comprehensive Metrics**: F1 score, recall, confusion matrix, and accuracy for thorough model evaluation

## Implementation Notes

- The `USE_BIAS_ON_LINEAR` flag in `fasterkan.py` controls whether linear layers use bias terms
- Dropout rate formula: `1-0.75**(num_grids)` ensures appropriate regularization as model complexity increases
- Gradient boosting factor of 10x for grid and inverse denominator parameters helps balance learning rates
