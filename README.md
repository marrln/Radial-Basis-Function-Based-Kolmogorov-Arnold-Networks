# RBF-based KANs on Image Classification - Radial Basis Function-Based Kolmogorov-Arnold Networks on Image Classification

This repository contains the implementation and exploration of **Kolmogorov-Arnold Networks (KANs)**, a neural network architecture inspired by the **Kolmogorov-Arnold representation theorem**. This theorem states that any multivariate continuous function can be represented as a finite superposition of univariate functions and a single two-variable function. 
The Kolmogorov-Arnold Networks leverage this powerful mathematical framework to create efficient and interpretable machine learning models.
For this specific implementation of the KANs we used Radial Basis Functions (RBF), hence RBF-based KANs, as learnable activations. In this particular repo we analyze the capabilities of RBF-based KANs on Image Classification.

---

### Prerequisites
To use this repository, ensure you have the following installed:
- Python 3.9.13
- Required libraries listed in `requirements.txt`

Install dependencies by running:
```bash
pip install -r requirements.txt
```

## Project Structure
```
kolmogorov-arnold-networks/
├── data/                  # Datasets for training and testing
├── configs/               # Configuration files
├── models/                # Neural network models
├── utils/                 # Utility scripts
├── train.py               # Training script
├── visualize.py           # Visualization script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---
## Contributing
Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request
---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## References
Pending ...

---
## Acknowledgments
Pending ...
