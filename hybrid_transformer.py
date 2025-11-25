"""
Hybrid Transformer with Memory Slots
=====================================

This module implements a transformer architecture with external memory slots
for long-context processing. It combines local windowed attention with 
cross-attention-based memory read/write operations.

Components:
-----------
- Core attention mechanisms (from NN assignment)
- Windowed attention for local context
- Cross-attention for memory reading
- Gated SSM for memory writing
- Hybrid transformer blocks and full model
- Needle-in-haystack dataset for testing

Usage:
------
    from hybrid_transformer import HybridTransformer, NeedleInHaystackDataset
    
    model = HybridTransformer(vocab_size=100, hidden_size=64, num_layers=2)
    dataset = NeedleInHaystackDataset(num_samples=1000, vocab_size=100)
"""

import math
import random

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# CORE ATTENTION MECHANISMS (From NN Assignment)
# ============================================================================

def scaled_dot_attention(q, k, v, mask=0):
    """
    Computes scaled dot product attention with an optional mask.
    
    Args:
        q: Query tensor of shape (batch, k, hidden_size)
        k: Key tensor of shape (batch, seq_len, hidden_size)
        v: Value tensor of shape (batch, seq_len, hidden_size)
        mask: Optional mask of shape (k, seq_len)
    
    Returns:
        context: Attention output of shape (batch, k, hidden_size)
        attention: Attention weights of shape (batch, k, seq_len)
    """
    unnorm_attn = torch.matmul(q, k.transpose(-2, -1))
    unnorm_attn = unnorm_attn / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
    masked_unnorm_attn = unnorm_attn + mask * -1e9
    attention = torch.softmax(masked_unnorm_attn, dim=-1)
    context = torch.matmul(attention, v)
    return context, attention


class Attention(nn.Module):
    """Standard attention mechanism for encoder-decoder models."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x, annots):
        """
        Args:
            x: Query input of shape (batch, k, hidden_size)
            annots: Key/value input of shape (batch, seq_len, hidden_size)
        """
        q = self.Q(x)
        k = self.K(annots)
        v = self.V(annots)
        return scaled_dot_attention(q, k, v)


class CausalAttention(Attention):
    """Causal (autoregressive) attention with causal masking."""
    
    def __init__(self, hidden_size):
        super().__init__(hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input of shape (batch, seq_len, hidden_size)
        """
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return scaled_dot_attention(q, k, v, mask)


# ============================================================================
# POSITIONAL ENCODING AND MLP (From NN Assignment)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, hidden_size, max_len=1000):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        dim = torch.arange(hidden_size // 2).float().unsqueeze(0)
        div_term = torch.exp(-math.log(10000.0) * (2 * dim) / hidden_size)
        angle = pos * div_term
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)
        self.register_buffer("pe", pe)

    def forward(self, idx):
        """
        Args:
            idx: Position indices
        Returns:
            Positional encodings for the given positions
        """
        return self.pe[idx]


class MLP(nn.Module):
    """Two-layer feedforward network with ReLU activation."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ============================================================================
# WINDOWED ATTENTION (New Component)
# ============================================================================

class WindowedAttention(nn.Module):
    """
    Local attention within a sliding window.
    Currently uses causal attention as placeholder - can be extended to 
    implement proper windowed masking.
    """
    
    def __init__(self, hidden_size, window_size=512):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.window_size = window_size
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch, seq_len, hidden_size)
        
        Returns:
            context: Attention output of shape (batch, seq_len, hidden_size)
            attention: Attention weights of shape (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        
        # TODO: Implement proper windowed attention mask
        # For now, using causal attention as placeholder
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        
        unnorm_attn = torch.matmul(q, k.transpose(-2, -1))
        unnorm_attn = unnorm_attn / math.sqrt(self.hidden_size)
        masked_unnorm_attn = unnorm_attn + mask * -1e9
        attention = torch.softmax(masked_unnorm_attn, dim=-1)
        context = torch.matmul(attention, v)
        
        return context, attention


# ============================================================================
# MEMORY COMPONENTS (New for Memory-Augmented Architecture)
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention for reading from external memory slots.
    Queries come from the input sequence, keys/values from memory.
    """
    
    def __init__(self, hidden_size, n_slots):
        super().__init__()
        self.Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.n_slots = n_slots
    
    def forward(self, x, memory):
        """
        Args:
            x: Input queries of shape (batch, seq_len, hidden_size)
            memory: Memory keys/values of shape (batch, n_slots, hidden_size)
        
        Returns:
            context: Retrieved memory context of shape (batch, seq_len, hidden_size)
        """
        q = self.Q(x)  # (batch, seq_len, hidden_size)
        k = self.K(memory)  # (batch, n_slots, hidden_size)
        v = self.V(memory)  # (batch, n_slots, hidden_size)
        
        context, attention = scaled_dot_attention(q, k, v, mask=0)
        return context


class GatedSSM(nn.Module):
    """
    Gated State Space Model for writing to memory.
    
    Implements the memory update equation:
        M_{t+1} = (1 - G_t) ⊙ M_t + G_t ⊙ φ(AM_t + BU_t)
    
    Where:
        - G_t is the gate (how much to update)
        - A is memory recurrence
        - B is input projection
        - φ is tanh activation
    """
    
    def __init__(self, hidden_size, n_slots):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        
        # Gate network: decides how much to update each memory slot
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Update parameters: what to write to memory
        self.A = nn.Linear(hidden_size, hidden_size, bias=False)  # Memory recurrence
        self.B = nn.Linear(hidden_size, hidden_size, bias=False)  # Input projection
        
    def forward(self, h, memory_state):
        """
        Args:
            h: Current hidden states of shape (batch, seq_len, hidden_size)
            memory_state: Current memory of shape (batch, n_slots, hidden_size)
        
        Returns:
            new_memory: Updated memory of shape (batch, n_slots, hidden_size)
        """
        # Summarize the sequence for memory update (simple mean pooling)
        summary = h.mean(dim=1, keepdim=True)  # (batch, 1, hidden_size)
        summary = summary.expand(-1, self.n_slots, -1)  # (batch, n_slots, hidden_size)
        
        # Compute gate: how much to update each memory slot
        gate = self.gate_net(summary)  # (batch, n_slots, hidden_size)
        
        # Compute update: what to write to memory
        update = torch.tanh(self.A(memory_state) + self.B(summary))
        
        # Blend old and new memory
        new_memory = (1 - gate) * memory_state + gate * update
        
        return new_memory


# ============================================================================
# HYBRID TRANSFORMER ARCHITECTURE
# ============================================================================

class HybridTransformerBlock(nn.Module):
    """
    Hybrid transformer block combining local attention with memory operations.
    
    Processing flow:
        1. Local windowed attention for nearby context
        2. Read from memory via cross-attention
        3. Standard MLP transformation
        4. Write to memory via gated SSM
    """
    
    def __init__(self, hidden_size, n_slots=16, window_size=512):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.norm3 = LayerNorm(hidden_size)
        
        # Local attention for nearby context
        self.local_attention = WindowedAttention(hidden_size, window_size)
        
        # Memory operations
        self.memory_read = CrossAttention(hidden_size, n_slots)
        self.memory_write = GatedSSM(hidden_size, n_slots)
        
        # Standard transformer MLP
        self.mlp = MLP(hidden_size)
        
    def forward(self, x, memory_state):
        """
        Args:
            x: Input of shape (batch, seq_len, hidden_size)
            memory_state: Memory of shape (batch, n_slots, hidden_size)
        
        Returns:
            output: Processed sequence of shape (batch, seq_len, hidden_size)
            new_memory: Updated memory of shape (batch, n_slots, hidden_size)
        """
        # 1. Local windowed attention
        h_local, _ = self.local_attention(self.norm1(x))
        h = x + h_local
        
        # 2. Read from memory using cross-attention
        context = self.memory_read(self.norm2(h), memory_state)
        h = h + context
        
        # 3. Standard MLP
        h = h + self.mlp(self.norm3(h))
        
        # 4. Write to memory
        new_memory = self.memory_write(h, memory_state)
        
        return h, new_memory


class HybridTransformer(nn.Module):
    """
    Complete hybrid transformer model with external memory slots.
    
    Architecture:
        - Token and positional embeddings
        - Learnable memory slot initialization
        - Stack of hybrid transformer blocks
        - Output projection head with weight tying
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Dimension of hidden representations
        num_layers: Number of transformer layers
        n_slots: Number of external memory slots
        window_size: Size of local attention window
    """
    
    def __init__(self, vocab_size, hidden_size, num_layers, n_slots=16, window_size=512):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size)
        
        # Initialize memory slots as learnable embeddings
        self.memory_init = nn.Parameter(torch.randn(1, n_slots, hidden_size) * 0.02)
        
        # Stack of hybrid blocks
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(hidden_size, n_slots, window_size) 
            for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying
        
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        
    def forward(self, inputs, memory_state=None):
        """
        Args:
            inputs: Token indices of shape (batch, seq_len)
            memory_state: Optional memory state of shape (batch, n_slots, hidden_size)
        
        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            final_memory: Final memory state of shape (batch, n_slots, hidden_size)
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        
        # Initialize memory if not provided
        if memory_state is None:
            memory_state = self.memory_init.expand(batch_size, -1, -1)
        
        # Embeddings
        pos = torch.arange(0, seq_len, device=inputs.device).long()
        x = self.tok_emb(inputs) + self.pos_emb(pos)
        
        # Pass through transformer blocks with memory
        for block in self.blocks:
            x, memory_state = block(x, memory_state)
        
        # Output projection
        logits = self.head(x)
        
        return logits, memory_state


# ============================================================================
# NEEDLE-IN-HAYSTACK DATASET
# ============================================================================

class NeedleInHaystackDataset(Dataset):
    """
    Toy dataset for testing long-context memory retrieval.
    
    Task: Find specific key-value pairs hidden in a long sequence of random tokens.
    
    Format:
        [haystack] <KEY> key_id <VALUE> value_id [haystack] ... <QUERY> key_id
    
    Goal: Predict the value associated with the queried key.
    
    This tests if the model can:
        1. Identify the needle (key-value pair) in a long sequence
        2. Store it in memory
        3. Retrieve it later when queried
    
    Args:
        num_samples: Number of examples in dataset
        vocab_size: Size of vocabulary (must be > num_needles * 2 + 3 for special tokens)
        haystack_length: Total sequence length
        num_needles: Number of key-value pairs to hide
        seed: Random seed for reproducibility
    """
    
    def __init__(self, num_samples=1000, vocab_size=100, 
                 haystack_length=1000, num_needles=5, seed=42):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.haystack_length = haystack_length
        self.num_needles = num_needles
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.KEY_TOKEN = vocab_size - 3
        self.VALUE_TOKEN = vocab_size - 2
        self.QUERY_TOKEN = vocab_size - 1
        
        # Key and value IDs (use tokens that won't be in haystack)
        self.key_ids = list(range(vocab_size - 3 - num_needles * 2, 
                                   vocab_size - 3 - num_needles))
        self.value_ids = list(range(vocab_size - 3 - num_needles, 
                                     vocab_size - 3))
        
        # Tokens available for haystack (excluding special tokens and key/value IDs)
        self.haystack_vocab = list(range(1, vocab_size - 3 - num_needles * 2))
        
        random.seed(seed)
        self.data = self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate all samples."""
        data = []
        for _ in range(self.num_samples):
            sample = self._generate_sample()
            data.append(sample)
        return data
    
    def _generate_sample(self):
        """
        Generate a single sample with format:
        [haystack] <KEY> key_id <VALUE> value_id [haystack] ... <QUERY> key_id -> target: value_id
        """
        # Create needle pairs (key-value associations)
        needle_pairs = list(zip(self.key_ids[:self.num_needles], 
                                self.value_ids[:self.num_needles]))
        random.shuffle(needle_pairs)
        
        # Choose which needle to query
        query_key, query_value = random.choice(needle_pairs)
        
        # Calculate space for haystack
        needle_tokens = self.num_needles * 4  # Each needle: <KEY> key_id <VALUE> value_id
        query_tokens = 2  # <QUERY> key_id
        available_haystack = self.haystack_length - needle_tokens - query_tokens
        
        # Distribute haystack tokens
        haystack_segments = []
        remaining = available_haystack
        for i in range(self.num_needles + 1):
            segment_length = remaining // (self.num_needles + 1 - i)
            haystack_segments.append(
                [random.choice(self.haystack_vocab) for _ in range(segment_length)]
            )
            remaining -= segment_length
        
        # Build sequence: interleave haystack and needles
        sequence = []
        for i, (key_id, value_id) in enumerate(needle_pairs):
            sequence.extend(haystack_segments[i])
            sequence.extend([self.KEY_TOKEN, key_id, self.VALUE_TOKEN, value_id])
        sequence.extend(haystack_segments[-1])
        
        # Add query at the end
        sequence.extend([self.QUERY_TOKEN, query_key])
        
        # Target is the value associated with the queried key
        target = query_value
        
        return {
            'input': torch.tensor(sequence, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'needle_pairs': needle_pairs,
            'query_key': query_key
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        """Custom collate function for DataLoader."""
        inputs = torch.stack([item['input'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        return inputs, targets


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def evaluate_random_baseline(dataset):
    """
    Evaluate random guessing performance on the needle-in-haystack task.
    
    Args:
        dataset: NeedleInHaystackDataset instance
    
    Returns:
        accuracy: Random baseline accuracy (percentage)
    """
    correct = 0
    total = len(dataset)
    
    for i in range(total):
        sample = dataset[i]
        random_guess = random.choice(dataset.value_ids)
        if random_guess == sample['target'].item():
            correct += 1
    
    accuracy = correct / total * 100
    return accuracy


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core attention
    'scaled_dot_attention',
    'Attention',
    'CausalAttention',
    
    # Basic components
    'PositionalEncoding',
    'MLP',
    
    # Windowed attention
    'WindowedAttention',
    
    # Memory components
    'CrossAttention',
    'GatedSSM',
    
    # Hybrid architecture
    'HybridTransformerBlock',
    'HybridTransformer',
    
    # Dataset
    'NeedleInHaystackDataset',
    
    # Utilities
    'evaluate_random_baseline',
    'count_parameters',
]
