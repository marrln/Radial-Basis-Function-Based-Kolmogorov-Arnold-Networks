# configs

This folder contains configuration scripts and utility modules for model training, evaluation, checkpointing, quantization, and plotting in the Kolmogorov-Arnold Networks (KANs) project.

## Contents

- **checkpoint.py**  
	Utilities for managing model checkpoints, hyperparameter-based directory structures, saving/loading model and optimizer states, and collecting model statistics.

- **custom_quant_fasterkan.py**  
	Implements custom quantization logic for FasterKAN models, including bit-width and data type management for model parameters.

- **fasterkan.py**  
	Core implementation of the FasterKAN model and its custom autograd functions.

- **mapper.py**  
	Maps string names to PyTorch optimizers, schedulers, and loss functions for dynamic configuration.

- **plotter.py**  
	Functions to plot training/validation accuracy and loss curves from JSON log files, with optional display of model hyperparameters.

- **training.py**  
	Training utilities, including model initialization, metric calculation, and integration with checkpointing and logging.

- **validate_custom_quant.py**  
	Scripts for validating custom quantization approaches, including dataset loading and evaluation routines.

- **fx_quant.ipynb**  
	Jupyter notebook for experimenting with quantization and model analysis.
