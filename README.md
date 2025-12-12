# Rolling State Memory (RSM) Transformer

Implementation of a Hybrid Transformer with Rolling State Memory for long-context language modeling.

**Course Project**: Deep Learning (Fall 2025)  
**Authors**: Daria Strait, Peter Vail Driscoll

## Overview

This project implements an RSM architecture that combines:
- **Local self-attention** within fixed-size windows
- **External memory slots** with gated state updates
- **Global synchronization** between tokens and memory
- **Efficient training** with truncated BPTT

### Key Features
- Modular architecture organized in 7 sections
- Multiple dataset support (TinyStories, Tiny Shakespeare, WikiText-103)
- Flexible tokenization (BPE, SentencePiece, character-level)
- Checkpoint management and experiment tracking
- Comprehensive evaluation suite (proposal sections 8 & 9)

## Quick Start

### Training a Model

```python
from hybrid_transformer1 import create_rsm_model, train_rsm_epoch

# Create model
model, global_sync, config = create_rsm_model(
    vocab_size=50257,
    hidden_size=256,
    num_layers=8,
    num_heads=4,
    num_memory_slots=32,
    chunk_size=512,
    use_global_sync=True,
    device='cuda'
)

# Train
metrics = train_rsm_epoch(
    model=model,
    global_sync=global_sync,
    data_iterator=train_loader,
    optimizer=optimizer,
    device='cuda'
)
```

### Running Evaluation

```python
from rsm_evaluation import evaluate_retrieval_at_distance, evaluate_cot_depth

# Test retrieval at different distances
retrieval_results = evaluate_retrieval_at_distance(
    model,
    distances=[128, 256, 512, 1024, 2048],
    device='cuda'
)

# Test chain-of-thought reasoning
cot_results = evaluate_cot_depth(model, max_depth=10, device='cuda')
```

See `run_evaluation.ipynb` for complete evaluation examples.

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
├── hybrid_transformer1.py        # Main RSM architecture (~940 lines)
│   ├── Section 1: NNDL Core (attention, MLP, etc.)
│   ├── Section 2: Memory Components (CrossAttention, GatedSSM, GlobalSync)
│   ├── Section 3: Architecture (HybridTransformerBlock, HybridTransformer)
│   ├── Section 4: Training Utilities
│   ├── Section 5: Generation Utilities
│   ├── Section 6: Dataset Utilities
│   └── Section 7: Helper Functions
│
├── rsm_evaluation.py             # Evaluation functions (NEW!)
│   ├── Retrieval at distance tests
│   ├── Chain-of-thought depth tests
│   ├── Ablation study functions
│   └── Memory telemetry analysis
│
├── run_evaluation.ipynb          # Evaluation experiments (NEW!)
│   ├── Run all evaluation metrics
│   ├── Generate plots and visualizations
│   └── Save results to JSON
│
├── test_rsmShakespeare.ipynb     # Training experiments
│   ├── Multiple dataset options
│   ├── Complete training pipeline
│   └── Text generation examples
│
├── EVALUATION_GUIDE.md           # Quick reference for evaluation code
└── README.md                     # This file
```

### Code Organization

- **Implementation code** → `.py` files (`hybrid_transformer1.py`, `rsm_evaluation.py`)
- **Experiments & plots** → `.ipynb` files (`run_evaluation.ipynb`, `test_rsmShakespeare.ipynb`)
- **This keeps notebooks clean** - just experiments, visualizations, and analysis

## Evaluation (Proposal Sections 8 & 9)

We implemented comprehensive evaluation metrics as specified in our proposal:

### 1. Retrieval at Distance
Tests exact-match recall for facts at varying distances (128-2048 tokens).

### 2. Chain-of-Thought Depth
Multi-step reasoning tasks to test how many steps the model can handle.

### 3. Ablation Studies
- Remove global sync → expect drops at multiples of K
- Freeze SSM writes → degradation on long-horizon tasks
- Vary m and chunk size → capacity vs cost curves

### 4. Memory Telemetry
- Linear decodability of slot contents
- Slot-level retention half-lives
- Cosine similarity tracking over time

All evaluation code is in `rsm_evaluation.py` with experiments in `run_evaluation.ipynb`.

## Training Example

See `test_rsmShakespeare.ipynb` for complete examples. Basic training loop:

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
## Implementation Notes (Proposal Section 9)

✅ **Short KV cache** - Using local causal attention only, no growing cache  
✅ **Salience pooling** - Both mean pooling and attention-based options  
✅ **A matrix initialization** - Near-identity with 0.9 decay for stability  
⚠️ **Shared memory per layer** - Future work

## Recent Updates

**December 11, 2025:**
- Fixed GatedSSM A matrix initialization (now near-identity with decay)
- Added attention-based salience pooling as alternative to mean pooling
- Created `rsm_evaluation.py` with all evaluation functions
- Created `run_evaluation.ipynb` for clean evaluation experiments
- Updated `create_rsm_model()` to support `use_attention_pooling` parameter

## References

- **TinyStories:** Eldan & Li (2023) - https://arxiv.org/abs/2305.07759
- **WikiText-103:** Merity et al. (2016) - https://arxiv.org/abs/1609.07843
- **Tiny Shakespeare:** Karpathy's char-rnn - https://github.com/karpathy/char-rnn

---

**Course**: Deep Learning (Fall 2025)  
**Authors**: Daria Strait, Peter Vail Driscoll  
**Repo**: [github.com/DariaZS/hybrid-transformer-rsm](https://github.com/DariaZS/hybrid-transformer-rsm)

**TinyStories (10 epochs, 256 hidden, 8 layers):**
- Training time: ~1-2 hours on CPU
- Parameters: ~13-15M (larger vocab)
- Expected accuracy: 50-60% (BPE token prediction)

## Citation

```bibtex
@misc{rsm2025,
  title={Rolling State Memory Transformer Implementation},
  author={Daria Strait, Peter Vail Driscoll},
  year={2025},
  note={Neural Network Deep Learning Course Project}
}
```

## References

- **TinyStories:** Eldan & Li (2023) - https://arxiv.org/abs/2305.07759
- **WikiText-103:** Merity et al. (2016) - https://arxiv.org/abs/1609.07843
- **Tiny Shakespeare:** Karpathy's char-rnn - https://github.com/karpathy/char-rnn
