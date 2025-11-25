# Example Usage of Hybrid Transformer Library

This demonstrates how to use the `hybrid_transformer.py` module.

## Quick Start

```python
# Import the library
from hybrid_transformer import (
    HybridTransformer,
    NeedleInHaystackDataset,
    evaluate_random_baseline,
    count_parameters
)
from torch.utils.data import DataLoader

# Create the model
model = HybridTransformer(
    vocab_size=100,
    hidden_size=64,
    num_layers=2,
    n_slots=8,
    window_size=128
)

print(f"Model has {count_parameters(model):,} parameters")

# Create dataset
dataset = NeedleInHaystackDataset(
    num_samples=1000,
    vocab_size=100,
    haystack_length=512,
    num_needles=5,
    seed=42
)

# Create dataloader
train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=dataset.collate_fn
)

# Test random baseline
random_acc = evaluate_random_baseline(dataset)
print(f"Random baseline: {random_acc:.2f}%")

# Forward pass
import torch
inputs = torch.randint(0, 100, (2, 50))
logits, memory = model(inputs)
print(f"Output shape: {logits.shape}")
print(f"Memory shape: {memory.shape}")
```

## Available Components

### Core Attention (from your NN assignment)
- `scaled_dot_attention()` - Basic attention function
- `Attention` - Standard attention
- `CausalAttention` - Autoregressive attention

### Basic Components (from your NN assignment)
- `PositionalEncoding` - Sinusoidal position embeddings
- `MLP` - Two-layer feedforward network

### New Components (for memory)
- `WindowedAttention` - Local attention within window
- `CrossAttention` - Read from memory slots
- `GatedSSM` - Write to memory slots

### Models
- `HybridTransformerBlock` - Single layer with memory
- `HybridTransformer` - Full model

### Dataset & Utils
- `NeedleInHaystackDataset` - Toy task for testing
- `evaluate_random_baseline()` - Baseline evaluation
- `count_parameters()` - Count model parameters

## File Structure

```
project/
├── hybrid_transformer.py      # Main library (this file)
├── test.ipynb                 # Your notebook
├── hw2_code_ds4286.ipynb     # Original assignment
└── README.md                  # This file
```
