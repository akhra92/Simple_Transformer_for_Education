import torch
from attention import SelfAttention
from transformer import SimpleTransformer
from positional_encoding import show_positional_encoding_pattern


def demo_self_attention():
    """Demonstrate self-attention mechanism."""
    print("\n" + "="*50)
    print("SELF-ATTENTION DEMO")
    print("="*50)
    
    # Create self-attention layer
    self_attention = SelfAttention(feature_size=5)
    
    # Create sample input
    sample_input = torch.randn(3, 5)  # 3 tokens, 5 features each
    print(f"Input shape: {sample_input.shape}")
    
    # Forward pass
    context, attention_weights = self_attention(sample_input)
    
    print(f"Context shape: {context.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights:\n{attention_weights}")


def demo_transformer():
    """Demonstrate full transformer on text example."""
    print("\n" + "="*50)
    print("TRANSFORMER DEMO")
    print("="*50)
    
    # Example text
    example_text = "I love Natural Language Processing"
    print(f"Example text: {example_text}")

    # Tokenize
    tokens = example_text.lower().split()
    print("Tokens:", tokens)

    # Create simple embeddings (in practice, you'd use learned embeddings)
    torch.manual_seed(0)
    vocab = {word: torch.randn(4) for word in tokens}
    print(f"Vocabulary size: {len(vocab)}")

    # Create embeddings tensor
    embeddings = torch.stack([vocab[token] for token in tokens])
    print("Embeddings shape:", embeddings.shape)
    print("'love' embedding:", vocab['love'])

    # Add batch dimension
    input_tensor = embeddings.unsqueeze(0)
    print("Input tensor shape:", input_tensor.shape)

    # Model parameters
    device = 'cpu'
    seq_len = len(tokens)
    d_model = 4
    num_heads = 2
    dim_ff = 8
    num_layers = 2

    # Create and run model
    model = SimpleTransformer(seq_len, d_model, num_heads, dim_ff, num_layers, device)
    
    print(f"\nModel parameters:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Feed-forward dimension: {dim_ff}")
    print(f"  Number of layers: {num_layers}")
    
    # Forward pass
    output, attention_weights = model(input_tensor, show_steps=True)
    
    print(f"\nFinal output shape: {output.shape}")
    print(f"Number of attention weight matrices: {len(attention_weights)}")
    print(f"Attention weights shape (layer 1): {attention_weights[0].shape}")


def main():
    """Run all demos."""
    print("Attention Mechanism and Transformer Demo")
    print("=" * 60)
    
    # Demo positional encoding
    print("\nRunning positional encoding demo...")
    pe = show_positional_encoding_pattern()
    
    # Demo self-attention
    demo_self_attention()
    
    # Demo full transformer
    demo_transformer()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()