import torch
import numpy as np

def transform_and_save_weights(pth_file, C):
    # Load the checkpoint 
    checkpoint = torch.load(pth_file, map_location='cpu', weights_only=True)
    model_state_dict = checkpoint['model_state_dict']

    # Find all layer.X.weight keys
    weight_keys = [k for k in model_state_dict if k.startswith('layers.') and (k.endswith('.weight') or k.endswith('.grid') or k.endswith('.inv_denom') ) ]
    # print(f"DEBUG: Found {len(weight_keys)} weight layers to process.")
    # print("DEBUG: Weight keys:", weight_keys)


    for key in weight_keys:
        
        W = model_state_dict[key].cpu().numpy()
        
        # Handle different tensor dimensions
        if len(W.shape) == 0:
            # NOTE: This is needed for the inv_denom tensors which are scalars
            # 0D tensor (scalar) - treat as 1x1 matrix
            M, N = 1, 1
            W = W.reshape(1, 1)
        elif len(W.shape) == 1: 
            # 1D tensor (vector) - treat as M x 1 matrix
            M = W.shape[0]
            N = 1
            W = W.reshape(M, 1)
        elif len(W.shape) == 2:
            # 2D tensor (matrix)
            M, N = W.shape
        else:
            print(f"Skipping {key}: unsupported shape {W.shape}")
            continue
        
        print(f" Matrix size: {M} x {N}, C: {C}")



        # Handle case where N < C (tensor is smaller than chunk size)
        if N < C:
            print(f"  N={N} < C={C}, using tensor as-is without chunking")
            W_prime = W
        else:
            # Normal chunking when N >= C
            num_chunks = N // C
            if num_chunks == 0:
                print(f"  Warning: num_chunks is 0, using tensor as-is")
                W_prime = W
            else:
                W_chunks = [W[:, i*C:(i+1)*C] for i in range(num_chunks)]
                
                # Handle remainder if N is not perfectly divisible by C
                remainder = N % C
                if remainder > 0:
                    W_chunks.append(W[:, num_chunks*C:])
                    print(f"  Added remainder chunk of size {remainder}")
                
                W_prime = np.concatenate(W_chunks, axis=1)
        
        W_flat = W_prime.flatten(order='C')

        # Save to binary file, one per layer per parameter type
        bin_file_prefix = key.split('.')[-1]
        layer_num = key.split('.')[1]
        bin_file = f"{bin_file_prefix}_layer{layer_num}.bin"
        with open(bin_file, 'wb') as f:
            f.write(W_flat.astype(np.int8).tobytes())
        print(f"Saved {key} to {bin_file}")

if __name__ == "__main__":
    pth_file = r"C:\Users\SOFINA\OneDrive - National and Kapodistrian University of Athens\KANs_\Training\Custom-Quantizer\mixed_16_bits\45482\BCELoss\Adam\ReduceOnPlateau\3.0e-05\[12288,1024,7]\[4]\-2.0e+00\2.5e-01\2.0e+00\epoch_best\model_checkpoint.pth"
    bin_file_prefix = "weights"
    C = 16  # adjust as needed

    transform_and_save_weights(pth_file, C)
