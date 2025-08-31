# Attention Mechanism and Transformer Implementation

A clean, educational implementation of attention mechanisms, positional encodings (including visualizations) and Transformer architecture using PyTorch.

## Features

- **Self-Attention Mechanism**: Implementation of scaled dot-product attention
- **Positional Encoding**: Sinusoidal positional encoding with visualization
- **Simple Transformer**: Multi-layer transformer encoder with multi-head attention
- **Interactive Demos**: Complete examples showing how to use each component

## Project Structure

```
Attention_Project/
├── attention.py              # Self-attention implementation
├── positional_encoding.py    # Positional encoding with visualization
├── transformer.py            # Complete transformer encoder
├── example.py               # Demo script with usage examples
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Attention_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Demo

To see all components in action:

```bash
python example.py
```

This will run three demos:
1. **Positional Encoding Visualization**: Shows 3D plots of positional encoding patterns
2. **Self-Attention Demo**: Demonstrates attention mechanism on sample data
3. **Full Transformer Demo**: Shows complete text processing pipeline

### Using Individual Components

#### Self-Attention

```python
from attention import SelfAttention
import torch

# Create self-attention layer
attention = SelfAttention(feature_size=512)

# Input: (batch_size, seq_len, feature_size)
input_tensor = torch.randn(1, 10, 512)

# Get contextualized representations and attention weights
context, attention_weights = attention(input_tensor)
```

#### Positional Encoding

```python
from positional_encoding import get_positional_encoding, show_positional_encoding_pattern

# Generate positional encoding
pe = get_positional_encoding(seq_len=100, d_model=512)

# Show visualization (optional)
show_positional_encoding_pattern()
```

#### Simple Transformer

```python
from transformer import SimpleTransformer
import torch

# Model parameters
seq_len = 50
d_model = 512
num_heads = 8
dim_ff = 2048
num_layers = 6
device = 'cpu'

# Create model
model = SimpleTransformer(seq_len, d_model, num_heads, dim_ff, num_layers, device)

# Forward pass
input_tensor = torch.randn(1, seq_len, d_model)
output, attention_weights = model(input_tensor)
```


## Dependencies

- PyTorch >= 1.9.0
- Matplotlib >= 3.5.0 (for visualizations)
- NumPy >= 1.21.0