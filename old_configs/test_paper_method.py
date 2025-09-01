import torch
import numpy as np

def test_transformations():
    # Create a simple test matrix
    M, N, C = 4, 6, 2
    W = torch.arange(M * N).reshape(M, N).float()
    
    print(f"Original matrix W ({M}×{N}):")
    print(W)
    print(f"Shape: {W.shape}")
    
    # Split into N/C matrices of C columns each
    split_matrices = torch.split(W, C, dim=1)
    print(f"\nSplit into {len(split_matrices)} matrices of {C} columns each:")
    for i, mat in enumerate(split_matrices):
        print(f"Matrix {i}: shape {mat.shape}")
        print(mat)
    
    # Method 1: Concatenate column-wise (as paper says)
    print("\n=== Method 1: Column-wise concatenation ===")
    try:
        result1 = torch.cat(split_matrices, dim=1)
        print(f"Result shape: {result1.shape}")
        print(result1)
        print("This just reconstructs the original matrix!")
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 2: Concatenate row-wise (as your code does)
    print("\n=== Method 2: Row-wise concatenation ===")
    result2 = torch.cat(split_matrices, dim=0)
    print(f"Result shape: {result2.shape}")
    print(result2)
    print(f"Expected shape for hardware: ({M}×{N//C}) × {C} = {M*N//C} × {C}")
    
    # Method 3: Stack the matrices (alternative interpretation)
    print("\n=== Method 3: Stack matrices ===")
    result3 = torch.stack(split_matrices, dim=0)
    print(f"Result shape: {result3.shape}")
    print(result3)

if __name__ == "__main__":
    test_transformations()
