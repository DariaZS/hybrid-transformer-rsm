# Evaluation Code Organization

## File Structure

```
project/
├── hybrid_transformer1.py      # Main architecture code
├── rsm_evaluation.py           # All evaluation functions (NEW!)
├── run_evaluation.ipynb        # Experiments & plots (NEW!)
├── test_rsmShakespeare.ipynb   # Training experiments
└── ...
```

## What's Where

### `rsm_evaluation.py` - Helper Functions
Contains all the evaluation logic:
- `evaluate_retrieval_at_distance()` - Test fact recall
- `evaluate_cot_depth()` - Multi-step reasoning
- `ablation_no_sync()` - Remove global sync test
- `ablation_freeze_ssm()` - Freeze memory writes test
- `ablation_vary_architecture()` - Test different configs
- `MemoryProbe` - Linear probe for memory analysis
- `analyze_memory_retention()` - Track memory decay
- `run_full_evaluation()` - Run everything at once

### `run_evaluation.ipynb` - Experiments Notebook
Just experiments and visualization:
- Load model
- Run each test
- Plot results
- Save figures

### `hybrid_transformer1.py` - Architecture
Main model code (already organized in 7 sections)

## How to Use

### Quick start:
```python
from rsm_evaluation import run_full_evaluation
from hybrid_transformer1 import create_rsm_model

# Create model
model, sync, config = create_rsm_model(vocab_size=1000, ...)

# Run all tests
results = run_full_evaluation(model, vocab_size=1000, device='cuda')
```

### Or test individual components:
```python
from rsm_evaluation import evaluate_retrieval_at_distance

results = evaluate_retrieval_at_distance(
    model,
    distances=[128, 256, 512],
    device='cuda'
)
```

## Key Improvements Made

1. **A matrix initialization** - Now uses near-identity with 0.9 decay
2. **Attention pooling option** - Can use attention instead of mean pooling
3. **All evaluation functions** - Moved to separate file for clarity
4. **Clean notebooks** - Just experiments, not implementation

## Implementation Notes (from Proposal Section 9)

✅ Short KV cache - Using local causal attention  
✅ Salience pooling - Both mean and attention-based options  
✅ A near identity - Initialized with 0.9 decay  
⚠️ Shared memory per layer - Future work
