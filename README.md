# RSM Transformer - Final Project

**Deep Learning Fall 2025**  
Daria Strait, Peter Vail Driscoll

Hybrid transformer with external memory for long-context modeling.

## Overview

We're combining standard transformer attention with external memory slots that can remember information from earlier in the sequence. The main idea is:

- Local attention works on chunks of text (like normal transformers)
- External memory stores important info across chunks
- Memory gets updated with a gated mechanism (similar to LSTMs)
- Periodic sync layer helps memory stay aligned with the sequence

This should help with tasks that need long-range dependencies.

## Usage

```python
from hybrid_transformer1 import create_rsm_model, train_rsm_epoch

# Create model
model, global_sync, config = create_rsm_model(
    vocab_size=65,     # for character-level
    hidden_size=256,
    num_layers=8,
    num_memory_slots=32,
    chunk_size=256,
    device='cuda'
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
metrics = train_rsm_epoch(model, global_sync, train_loader, optimizer, device='cuda')
```

Check out `test_rsmShakespeare.ipynb` for the full training setup.

## Code Structure

`hybrid_transformer1.py` has everything organized into sections:

1. Core functions (attention, MLP, etc.) - reused from HW2
2. Memory components (CrossAttention, GatedSSM, GlobalSync)
3. Main architecture (blocks and full model)
4. Training utilities
5. Text generation
6. Dataset handling
7. Helper functions

## Datasets

- **Tiny Shakespeare** - character-level (65 tokens), quick to train
- **TinyStories** - BPE tokens (50K vocab), more realistic benchmark
- **WikiText-103** - haven't tested this yet but it's set up

## Files

- `hybrid_transformer1.py` - main model code (~940 lines)
- `rsm_evaluation.py` - evaluation functions (retrieval, ablations, etc.)
- `run_evaluation.ipynb` - runs all the tests
- `test_rsmShakespeare.ipynb` - training notebook
- `EVALUATION_GUIDE.md` - explains the evaluation code
- `requirements.txt` - dependencies

We kept implementation in .py files and experiments in notebooks to keep things organized.

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
