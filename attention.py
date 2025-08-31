import torch
from torch.nn import *


class SelfAttention(Module):
    """
    Self-attention mechanism implementation.
    
    Args:
        feature_size (int): Size of input features
    """

    def __init__(self, feature_size):
        super().__init__()
        self.qry_layer = Linear(in_features=feature_size, out_features=feature_size)
        self.key_layer = Linear(in_features=feature_size, out_features=feature_size)
        self.val_layer = Linear(in_features=feature_size, out_features=feature_size)

    def forward(self, inp):
        """
        Forward pass of self-attention mechanism.
        
        Args:
            inp (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_size)
            
        Returns:
            tuple: (context, attention_weights)
                - context: Contextualized representations
                - attention_weights: Attention weight matrix
        """
        
        Q = self.qry_layer(inp)
        K = self.key_layer(inp)
        V = self.val_layer(inp)        

        scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(inp.size(-1), dtype=torch.float32))
        attention_weights = functional.softmax(scores, dim=1)
        context = torch.matmul(attention_weights, V)

        return context, attention_weights