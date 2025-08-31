import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def show_positional_encoding_pattern():
    """
    Demonstrates positional encoding patterns used in Transformer models.
    
    Returns:
        torch.Tensor: Positional encoding matrix of shape (seq_length, d_model)
    """
    print(f"\n" + "="*60)
    print("POSITIONAL ENCODING PATTERN")
    print("="*60)
    
    d_model = 64
    seq_length = 100
    
    # Calculate div_term
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-math.log(10000.0) / d_model))
    
    print(f"d_model = {d_model}")
    print(f"seq_length = {seq_length}")
    print(f"Number of frequency pairs: {len(div_term)}")
    
    # Create positional encoding matrix
    pe = torch.zeros(seq_length, d_model)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    
    # Calculate angles for each position and dimension
    angles = position * div_term
    
    # Apply sin/cos
    pe[:, 0::2] = torch.sin(angles)  # Even dimensions
    pe[:, 1::2] = torch.cos(angles)  # Odd dimensions
    
    print(f"Positional Encoding Matrix Shape: {pe.shape}")
    
    # Create comprehensive visualization
    create_positional_encoding_plot(pe, d_model, seq_length)
    
    return pe


def create_positional_encoding_plot(pe, d_model, seq_length):
    """
    Create 3D plot to visualize positional encoding patterns.
    
    Args:
        pe (torch.Tensor): Positional encoding matrix
        d_model (int): Model dimension
        seq_length (int): Sequence length
    """
    
    # Convert to numpy for plotting
    pe_np = pe.numpy()
    positions = np.arange(seq_length)
    dims_3d = min(3, d_model)
    pos_sample = positions[::5]  # Sample positions for clarity
    pe_sample = pe_np[::5]

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(12, 8))    
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(dims_3d):
        ax.plot(pos_sample, [i]*len(pos_sample), pe_sample[:, i], 
                label=f'dim{i}', linewidth=2) 
    
    ax.set_xlabel('Position')      
    ax.set_ylabel('Dimension')     
    ax.set_zlabel('Encoding Value') 
    ax.set_title('3D View of First 5 Dimensions')
    ax.legend()
    
    plt.show()


def get_positional_encoding(seq_len, d_model, device='cpu'):
    """
    Generate positional encoding for transformer models.
    
    Args:
        seq_len (int): Sequence length
        d_model (int): Model dimension
        device (str): Device to place tensor on
        
    Returns:
        torch.Tensor: Positional encoding matrix
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)        
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -math.log(10000) / d_model)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


if __name__ == "__main__":    
    pe = show_positional_encoding_pattern()