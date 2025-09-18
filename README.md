# Radial Basis Function-based Kolmogorov-Arnold Networks (RBF-KAN)

This repository implements Radial Basis Function (RBF)-based Kolmogorov-Arnold Networks (KAN), a neural network architecture based on the Kolmogorov-Arnold representation theorem. The design (called [FasterKAN](https://github.com/AthanasiosDelis/faster-kan)) includes specific optimizations for both training and potential hardware acceleration.

This repository focuses on image classification, with examples on handwritten digit recognition (MNIST) and medical image classification (HAM10000 skin lesions). It is based on the original [FasterKAN](https://github.com/AthanasiosDelis/faster-kan) with several key modifications for improved performance and hardware deployability.

## Overview

KANs are neural networks inspired by the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be represented as a composition of continuous functions of a single variable and addition operations. RBF-KANs use radial basis functions as the inner univariate functions, providing a flexible framework for function approximation.

While originally conceived for general function approximation, our implementation focuses specifically on image classification tasks. The architecture processes flattened image data and has been optimized for classification performance on image datasets.

This implementation includes:
- Core RBF-KAN architecture (based on [FasterKAN](https://github.com/AthanasiosDelis/faster-kan))
- Quantization support for deployment on resource-constrained devices
- Example applications for MNIST digit classification and HAM10000 skin lesion classification

## Key Features

- **RBF-based Architecture for Image Classification**: Uses radial basis functions in a KAN architecture specifically tuned for image classification tasks
- **Training Optimizations**: 
  - Dropout scaled with grid size for better generalization
  - Gradient boosting for improved grid and denominator training
  - Configurable biases for FPGA-friendly implementation
- **Quantization Support**: Tools for post-training quantization
- **Image Classification Examples**: Ready-to-use examples on standard image datasets (MNIST, HAM10000)

## Differences from FasterKAN

Our implementation includes several key modifications to the original [FasterKAN](https://github.com/AthanasiosDelis/faster-kan):

1. **Dropout Layers**: Added dropout layers with rates that scale with the number of grid points to improve generalization, especially for models with larger grid sizes.
   
2. **Hardware-Friendly Architecture**: Optional bias term on linear layers, making the model more suitable for acceleration on FPGA devices.

3. **Modified Backward Pass**: Enhanced the backward pass with gradient boosting for grid and inverse denominator parameters for improved training effort and enhanced metrics performance.

4. **Quantization Support**: Added comprehensive tools for model quantization to support deployment on resource-constrained devices.

## Repository Structure

- **`kan_utils/`**: Core implementation of KAN architectures and utilities
  - `fasterkan.py`: RBF-KAN implementation (FasterKAN) with dropout and gradient boosting optimizations
  - `mapper.py`: Utility for mapping string identifiers to PyTorch optimizer, scheduler, and loss function classes
  - `training.py`: Comprehensive training pipeline with initialization, logging, and metrics computation
  - `checkpoint_utils.py`: Functions for model checkpointing, config management, and model serialization
  - `experiment_eval.py`: Tools for experiment analysis and hyperparameter management
  - `plotter.py`: Visualization utilities for training metrics and model evaluation
  - `general_utils.py`: Common utility functions shared across the codebase
  - `quant_fasterkan.py`: Custom quantization implementation of the float model `fasterkan.py` and helpful functions for quantization

- **`MNIST/`**: MNIST dataset experiments
  - `mnist_kan_training.ipynb`: Complete training pipeline notebook for MNIST classification
  - `mnist_kan_quant.ipynb`: Model quantization experiments with performance benchmarks

- **`HAM1000/`**: Skin cancer classification using HAM10000 dataset
  - `SkinCancerDataset.py`: Custom dataset implementation for dermoscopic images
  - `config.json`: Configuration parameters for HAM10000 experiments
  - `HAM1000_kan_training.ipynb`: Complete training pipeline notebook for HAM1000 classification
  - `HAM1000_kan_quant.ipynb`: Model quantization experiments with performance benchmarks
  - `README.md`: Detailed information about the HAM10000 dataset and its use with KANs

- **`quantization/`**: Specialized quantization tools and examples
  - `custom_quant_fasterkan.py`: Fixed-point quantization implementation for FasterKAN models
  - `fx_quant.ipynb`: PyTorch FX-based quantization examples and benchmarks
  - `validate_custom_quant.py`: Validation tools for assessing quantized model performance

### Training a KAN Model

See the `MNIST/mnist_kan_training.ipynb` notebook for a complete training pipeline on the MNIST dataset.
See the `HAM10000/HAM10000_kan_training.ipynb` notebook for a complete training pipeline on the HAM10000 dataset.


### Quantizing a KAN Model

See the 'MNIST/mnist_kan_quant.ipynb' notebook for quantization examples on MNIST.
See the 'HAM10000/HAM10000_kan_quant.ipynb' notebook for quantization examples on HAM10000.


## HAM10000 Dataset

This repository includes code for applying KANs to the HAM10000 dataset, a collection of dermoscopic images for skin cancer classification. The dataset contains 10,015 images across seven categories of skin lesions. 
For more details, see the `HAM10000/README.md` file.

## Differences from Traditional Neural Networks

RBF-KANs differ from traditional neural networks in several ways:
1. Based on the Kolmogorov-Arnold representation theorem
2. Use radial basis functions instead of typical activation functions
3. Specific architecture with grid-based operations
4. Optimized for function approximation
5. Despite lacking spacial locality (like CNNs), they can still effectively process image data

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This research is part of the [kan-fpga](https://github.com/gvenit/kan-fpga.git) project, an open-source soft IP core for RBF-KAN acceleration on SoC devices.

## References

- [1] Liu, Z.; Wang, Y.; Vaidya, S.; Ruehle, F.; Halverson, J.; Soljaˇci´c, M.; Hou, T.Y.; Tegmark, M. KAN: Kolmogorov-Arnold Networks 2024. [[arXiv:cs.LG/2404.19756](http://arxiv.org/abs/2404.19756)].
- [2] Li, Z. Kolmogorov-Arnold Networks are Radial Basis Function Networks 2024. [[arXiv:cs.LG/2405.06721](http://arxiv.org/abs/2405.06721)]. 
- [3] Delis, A. FasterKAN. https://github.com/AthanasiosDelis/faster-kan/, 2024.
- [4] Tschandl, P.; Rosendahl, C.; Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 2018, 5, 180161. DOI : [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161).
