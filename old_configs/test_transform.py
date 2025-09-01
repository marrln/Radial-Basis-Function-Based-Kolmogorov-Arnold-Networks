import torch
import numpy as np

# Test the transformation with a simple example
def transform_linear_weights(linear_weights, C):
    M, N = linear_weights.shape
    if N % C != 0:
        padding_size = C - (N % C)
        padding = torch.zeros((M, padding_size), dtype=linear_weights.dtype)
        linear_weights = torch.cat((linear_weights, padding), dim=1)
        N += padding_size
    
    split_matrices = torch.split(linear_weights, C, dim=1)
    transformed = torch.cat(split_matrices, dim=0)
    return transformed

# Test with a 4x6 matrix and C=2
test_matrix = torch.arange(24).reshape(4, 6).float()
print('Original matrix (4x6):')
print(test_matrix)
print('Shape:', test_matrix.shape)

transformed = transform_linear_weights(test_matrix, 2)
print('\nTransformed matrix:')
print(transformed)
print('Shape:', transformed.shape)
print('Expected shape: (4*6/2, 2) = (12, 2)')
