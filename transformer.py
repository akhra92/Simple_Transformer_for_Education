import torch
import math
from torch.nn import *
from positional_encoding import get_positional_encoding


class SimpleTransformer(Module):
    """
    A simple Transformer encoder implementation.
    
    Args:
        seq_len (int): Maximum sequence length
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        dim_ff (int): Feed-forward network dimension
        num_layers (int): Number of encoder layers
        device (str): Device to place model on
    """

    def __init__(self, seq_len, d_model, num_heads, dim_ff, num_layers, device):
        super().__init__()

        self.seq_len, self.d_model = seq_len, d_model
        self.device = device

        self.pos_enc = get_positional_encoding(seq_len, d_model, device)

        self.layers = ModuleList([self.EncoderLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)])

    class EncoderLayer(Module):
        """
        Single Transformer encoder layer with multi-head attention and feed-forward network.
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            dim_ff (int): Feed-forward network dimension
        """
        def __init__(self, d_model, num_heads, dim_ff):
            super().__init__()

            self.attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
            self.ff = Sequential(Linear(d_model, dim_ff), ReLU(), Linear(dim_ff, d_model))

            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)

        def forward(self, inp):
            """
            Forward pass through encoder layer.
            
            Args:
                inp (torch.Tensor): Input tensor
                
            Returns:
                tuple: (output, attention_weights)
            """

            attn_out, attn_weights = self.attn(inp, inp, inp)
            inp = self.norm1(inp + attn_out)
            ff_out = self.ff(inp)
            inp = self.norm2(inp + ff_out)

            return inp, attn_weights
        
    def forward(self, inp, show_steps=True):
        """
        Forward pass through the Transformer.
        
        Args:
            inp (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            show_steps (bool): Whether to print intermediate shapes
            
        Returns:
            tuple: (output, all_attention_weights)
        """

        if show_steps: 
            print(f"input shape -> {inp.shape}")

        inp = inp + self.pos_enc.unsqueeze(0)

        if show_steps: 
            print(f"after pos encoding shape -> {inp.shape}")

        all_attn_weights = []

        for i, layer in enumerate(self.layers):
            inp, attn_weights = layer(inp)
            all_attn_weights.append(attn_weights)        
            if show_steps:
                print(f"inp After layer {i+1} -> {inp.shape}")
                print(f"attn_weights After layer {i+1} -> {attn_weights.shape}")

        return inp, all_attn_weights