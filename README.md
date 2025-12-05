# Rolling State Memory (RSM) Transformer

A modular implementation of a Hybrid Transformer with Rolling State Memory for long-context language modeling.

## Overview

This project implements an RSM architecture that combines:
- **Local self-attention** within fixed-size windows
- **External memory slots** with gated state updates
- **Global synchronization** between tokens and memory
- **Efficient training** with truncated BPTT

**Key Features:**
- 🎯 Modular, clean architecture with 7 organized sections
- 📚 Multiple dataset support (TinyStories, Tiny Shakespeare, WikiText-103)
- 🔤 Flexible tokenization (BPE, SentencePiece, character-level)
- 💾 Checkpoint management and experiment tracking
- 🎨 Built-in visualization and text generation

## Quick Start

```python
from hybrid_transformer1 import create_rsm_model, train_rsm_epoch, generate_with_rsm

# Create model (easy factory function)
model, global_sync, config = create_rsm_model(
    vocab_size=50257,      # GPT-2 tokenizer
    hidden_size=256,       # Model dimension
    num_layers=8,          # Transformer layers
    num_heads=4,           # Attention heads
    num_memory_slots=32,   # External memory size
    chunk_size=512,        # Context window
    dropout=0.1,
    use_global_sync=True,
    device='cuda'
)

# Train for one epoch
metrics = train_rsm_epoch(
    model=model,
    global_sync=global_sync,
    data_iterator=train_loader,
    optimizer=optimizer,
    device='cuda'
)

# Generate text
generated_tokens = generate_with_rsm(
    model=model,
    prompt_tokens=[1, 2, 3],
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)
```

## Architecture Components

### Section 1: NNDL Core Functions
From Neural Network Deep Learning assignment - foundational building blocks:
- `scaled_dot_attention()` - Scaled dot-product attention
- `PositionalEncoding` - Sinusoidal position embeddings
- `MLP` - Feed-forward network with configurable expansion
- `CausalSelfAttention` - Multi-head causal attention

### Section 2: Memory Components
External memory mechanisms for long-context processing:
- `CrossAttention` - Read from external memory slots
- `GatedSSM` - Gated state space model for memory updates
- `GlobalSyncLayer` - Bidirectional token-memory synchronization

### Section 3: Architecture
Complete transformer architecture:
- `HybridTransformerBlock` - Single layer with local attention + memory
- `HybridTransformer` - Full model with learnable memory initialization

### Section 4: Training Utilities
- `train_rsm_epoch()` - Efficient training with truncated BPTT
- Gradient clipping, global sync cadence, memory reset options

### Section 5: Generation Utilities
- `generate_with_rsm()` - Autoregressive text generation
- Temperature sampling, top-k, nucleus (top-p) filtering

### Section 6: Dataset Utilities
- `ChunkedSequenceDataset` - Overlapping sequence chunks for training
- Compatible with any tokenization scheme

### Section 7: Helper Functions
- `create_rsm_model()` - Factory function with sensible defaults
- `save_checkpoint()` / `load_checkpoint()` - Model persistence
- `count_parameters()` - Count trainable parameters

## Datasets

### TinyStories (Default)
- **Paper:** Eldan & Li (2023) - "TinyStories: How Small Can Language Models Be..."
- **Size:** 2.1M synthetic stories (~250K tokens for 500 stories)
- **Tokenization:** GPT-2 BPE (50K vocab)
- **Use case:** Standard small LM benchmark

### Tiny Shakespeare (Fast Alternative)
- **Source:** Complete works of Shakespeare
- **Size:** ~1MB text, ~1M characters
- **Tokenization:** Character-level (65 vocab)
- **Use case:** Quick experiments, 5-10x faster training

### WikiText-103
- **Paper:** Merity et al. (2016) - "Pointer Sentinel Mixture Models"
- **Size:** 103M tokens from Wikipedia
- **Use case:** Long-form text benchmark

## File Structure

```
project/
├── hybrid_transformer1.py     # Main RSM implementation (~900 lines)
│   ├── Section 1: NNDL Core
│   ├── Section 2: Memory Components
│   ├── Section 3: Architecture
│   ├── Section 4: Training
│   ├── Section 5: Generation
│   ├── Section 6: Dataset
│   └── Section 7: Helpers
│
├── test_rsm1.ipynb            # Experimental notebook
│   ├── Multiple dataset options
│   ├── Complete training pipeline
│   ├── Visualization & evaluation
│   └── Text generation examples
│
├── hw2_code_ds4286.ipynb      # Original NN assignment
└── README.md                  # This file
```

## Training Example

See `test_rsm1.ipynb` for complete examples. Basic training loop:

```python
import torch
from hybrid_transformer1 import *

# Load data (example: Tiny Shakespeare)
import requests
text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
all_tokens = [char_to_idx[ch] for ch in text]

# Create dataset
dataset = ChunkedSequenceDataset(tokens=all_tokens, chunk_size=256)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model, global_sync, config = create_rsm_model(
    vocab_size=len(chars),
    hidden_size=256,
    num_layers=8,
    num_heads=4,
    num_memory_slots=32,
    chunk_size=512,
    device='cuda'
)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for epoch in range(10):
    metrics = train_rsm_epoch(model, global_sync, loader, optimizer, device='cuda')
    print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']*100:.2f}%")
```

## Performance

**Tiny Shakespeare (10 epochs, 256 hidden, 8 layers):**
- Training time: ~15-30 minutes on CPU
- Parameters: ~5-8M
- Expected accuracy: 40-50% (character-level prediction)

**TinyStories (10 epochs, 256 hidden, 8 layers):**
- Training time: ~1-2 hours on CPU
- Parameters: ~13-15M (larger vocab)
- Expected accuracy: 50-60% (BPE token prediction)


## References

- **TinyStories:** Eldan & Li (2023) - https://arxiv.org/abs/2305.07759
- **WikiText-103:** Merity et al. (2016) - https://arxiv.org/abs/1609.07843
- **Tiny Shakespeare:** Karpathy's char-rnn - https://github.com/karpathy/char-rnn
