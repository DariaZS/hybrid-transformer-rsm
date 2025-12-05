"""
Rolling State Memory (RSM) Architecture - Professional Implementation
======================================================================

A clean, modular implementation of the Hybrid Transformer with Rolling State Memory.
Organized into logical sections for easy navigation and maintenance.

Sections:
---------
1. NNDL Core Functions (from NN Deep Learning assignment)
2. Memory Components (Cross-attention, Gated SSM)
3. Architecture (Hybrid Transformer blocks and model)
4. Training Utilities (Generic training loop)
5. Generation Utilities (Autoregressive generation)
6. Dataset Utilities (Flexible data loading)
7. Helper Functions (Checkpoint, metrics, etc.)

Author: Daria Strait, Peter Vail Driscoll
Date: December 2025
"""

import math
import random
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================================
# SECTION 1: NNDL CORE FUNCTIONS
# Neural Network Deep Learning Assignment - Foundational Components
# ============================================================================

def scaled_dot_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                        mask: torch.Tensor = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention mechanism.
    
    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        q: Query tensor (batch, k, hidden_size)
        k: Key tensor (batch, seq_len, hidden_size)
        v: Value tensor (batch, seq_len, hidden_size)
        mask: Optional mask (k, seq_len) - 1 for masked positions
    
    Returns:
        context: Attention output (batch, k, hidden_size)
        attention: Attention weights (batch, k, seq_len)
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if isinstance(mask, torch.Tensor):
        scores = scores + mask * -1e9
    
    attention = F.softmax(scores, dim=-1)
    context = torch.matmul(attention, v)
    
    return context, attention


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                            (-math.log(10000.0) / hidden_size))
        
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Position indices (seq_len,) or (batch, seq_len)
        
        Returns:
            Positional encodings
        """
        return self.pe[positions]


class MLP(nn.Module):
    """
    Two-layer feedforward network with configurable expansion ratio.
    
    Standard in transformers: hidden -> 4*hidden -> hidden
    """
    
    def __init__(self, hidden_size: int, expansion_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * expansion_ratio)
        self.fc2 = nn.Linear(hidden_size * expansion_ratio, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with optional multi-head support.
    Prevents attending to future positions (autoregressive).
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask cache
        self.register_buffer('causal_mask', None, persistent=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (batch, seq_len, hidden_size)
        
        Returns:
            output: Attention output (batch, seq_len, hidden_size)
            attention: Attention weights (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create or retrieve causal mask
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self.causal_mask = mask.to(x.device)
        else:
            mask = self.causal_mask[:seq_len, :seq_len]
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_size)
        out = self.proj(out)
        
        return out, attention


# ============================================================================
# SECTION 2: MEMORY COMPONENTS
# External Memory Mechanisms for Long-Context Processing
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention for reading from external memory.
    
    Queries from input sequence, keys/values from memory slots.
    This allows the model to retrieve relevant information stored in memory.
    """
    
    def __init__(self, hidden_size: int, n_slots: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Queries (batch, seq_len, hidden_size)
            memory: Keys/values (batch, n_slots, hidden_size)
        
        Returns:
            context: Retrieved information (batch, seq_len, hidden_size)
        """
        q = self.q_proj(x)
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        
        context, _ = scaled_dot_attention(q, k, v, mask=0)
        context = self.out_proj(context)
        context = self.dropout(context)
        
        return context


class GatedSSM(nn.Module):
    """
    Gated State Space Model for memory updates.
    
    Implements: M_{t+1} = (1 - G) ⊙ M_t + G ⊙ tanh(A*M_t + B*U_t)
    
    Where:
        - G: Gate (what to update)
        - A: Memory recurrence matrix
        - B: Input projection matrix
        - U_t: Update signal from current tokens
    """
    
    def __init__(self, hidden_size: int, n_slots: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        
        # Gate network: controls update magnitude
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # SSM parameters
        self.A = nn.Linear(hidden_size, hidden_size, bias=False)  # Memory recurrence
        self.B = nn.Linear(hidden_size, hidden_size, bias=False)  # Input projection
    
    def forward(self, h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Current hidden states (batch, seq_len, hidden_size)
            memory: Current memory state (batch, n_slots, hidden_size)
        
        Returns:
            new_memory: Updated memory (batch, n_slots, hidden_size)
        """
        # Aggregate sequence information (mean pooling)
        summary = h.mean(dim=1, keepdim=True)  # (batch, 1, hidden_size)
        summary = summary.expand(-1, self.n_slots, -1)  # (batch, n_slots, hidden_size)
        
        # Compute gate and update
        gate = self.gate_net(summary)
        update = torch.tanh(self.A(memory) + self.B(summary))
        
        # Gated update
        new_memory = (1 - gate) * memory + gate * update
        
        return new_memory


class GlobalSyncLayer(nn.Module):
    """
    Bidirectional synchronization between tokens and memory.
    
    Performed periodically (every K chunks) to:
    1. Aggregate token information into memory (tokens → memory)
    2. Broadcast memory information to tokens (memory → tokens)
    
    Prevents memory drift over very long sequences.
    """
    
    def __init__(self, hidden_size: int, n_slots: int):
        super().__init__()
        
        # Memory reads from tokens
        self.mem_from_tokens = CrossAttention(hidden_size, n_slots)
        
        # Tokens read from memory
        self.tokens_from_mem = CrossAttention(hidden_size, n_slots)
        
        # Layer normalization
        self.norm_tokens = nn.LayerNorm(hidden_size)
        self.norm_memory = nn.LayerNorm(hidden_size)
    
    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: Token representations (batch, seq_len, hidden_size)
            memory: Memory state (batch, n_slots, hidden_size)
        
        Returns:
            synced_tokens: Updated tokens (batch, seq_len, hidden_size)
            synced_memory: Updated memory (batch, n_slots, hidden_size)
        """
        # Step 1: Memory aggregates from tokens
        mem_update = self.mem_from_tokens(self.norm_memory(memory), tokens)
        synced_memory = memory + mem_update
        
        # Step 2: Tokens read from updated memory
        token_update = self.tokens_from_mem(self.norm_tokens(tokens), synced_memory)
        synced_tokens = tokens + token_update
        
        return synced_tokens, synced_memory


# ============================================================================
# SECTION 3: ARCHITECTURE
# Hybrid Transformer with Rolling State Memory
# ============================================================================

class HybridTransformerBlock(nn.Module):
    """
    Single transformer block with memory integration.
    
    Flow:
        1. Local self-attention (within-chunk attention)
        2. Memory read (cross-attention from memory)
        3. Feedforward network (MLP)
        4. Memory write (update memory with gated SSM)
    """
    
    def __init__(self, hidden_size: int, n_slots: int = 32, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # Local attention
        self.self_attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        
        # Memory operations
        self.memory_read = CrossAttention(hidden_size, n_slots, dropout)
        self.memory_write = GatedSSM(hidden_size, n_slots)
        
        # Feedforward
        self.mlp = MLP(hidden_size, expansion_ratio=4, dropout=dropout)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens (batch, seq_len, hidden_size)
            memory: Memory state (batch, n_slots, hidden_size)
        
        Returns:
            output: Processed tokens (batch, seq_len, hidden_size)
            new_memory: Updated memory (batch, n_slots, hidden_size)
        """
        # 1. Local self-attention
        attn_out, _ = self.self_attn(self.norm1(x))
        x = x + attn_out
        
        # 2. Read from memory
        mem_out = self.memory_read(self.norm2(x), memory)
        x = x + mem_out
        
        # 3. Feedforward
        x = x + self.mlp(self.norm3(x))
        
        # 4. Update memory
        new_memory = self.memory_write(x, memory)
        
        return x, new_memory


class HybridTransformer(nn.Module):
    """
    Complete Hybrid Transformer with Rolling State Memory.
    
    Key features:
        - Token and positional embeddings
        - Learnable memory initialization
        - Stack of hybrid transformer blocks
        - Weight tying between embedding and output projection
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 n_slots: int = 32, window_size: int = 256, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_slots = n_slots
        self.window_size = window_size
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size, max_len=window_size * 2)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable memory initialization
        self.memory_init = nn.Parameter(torch.randn(1, n_slots, hidden_size) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(hidden_size, n_slots, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input token IDs (batch, seq_len)
            memory: Optional memory state (batch, n_slots, hidden_size)
        
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            final_memory: Final memory state (batch, n_slots, hidden_size)
        """
        batch_size, seq_len = x.shape
        
        # Initialize memory if not provided
        if memory is None:
            memory = self.memory_init.expand(batch_size, -1, -1)
        
        # Embeddings
        pos = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)
        
        # Pass through transformer blocks
        for block in self.blocks:
            h, memory = block(h, memory)
        
        # Output projection
        h = self.norm_out(h)
        logits = self.head(h)
        
        return logits, memory


# ============================================================================
# SECTION 4: TRAINING UTILITIES
# Generic training functions for RSM
# ============================================================================

def train_rsm_epoch(
    model: HybridTransformer,
    global_sync: Optional[GlobalSyncLayer],
    data_iterator,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sync_cadence: int = 4,
    max_grad_norm: float = 1.0,
    reset_memory_every: Optional[int] = None
) -> Dict[str, float]:
    """
    Train RSM for one epoch with truncated BPTT.
    
    Args:
        model: HybridTransformer model
        global_sync: Optional GlobalSyncLayer for periodic synchronization
        data_iterator: Iterator yielding (x, y) or (x, y, metadata) tuples
        optimizer: PyTorch optimizer
        device: Device to train on
        sync_cadence: Perform global sync every K chunks
        max_grad_norm: Gradient clipping threshold
        reset_memory_every: Reset memory every N chunks (None = never)
    
    Returns:
        metrics: Dictionary with 'loss', 'accuracy', 'num_chunks'
    """
    model.train()
    if global_sync is not None:
        global_sync.train()
    
    # Initialize memory
    batch_size = 1
    memory = model.memory_init.expand(batch_size, -1, -1).detach()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_chunks = 0
    
    for chunk_idx, batch in enumerate(data_iterator):
        # Flexible unpacking
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        
        # Ensure proper shape and device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        x = x.to(device)
        y = y.to(device)
        
        # Optional memory reset
        if reset_memory_every and chunk_idx % reset_memory_every == 0 and chunk_idx > 0:
            memory = model.memory_init.expand(batch_size, -1, -1).detach()
        
        # Forward pass
        logits, new_memory = model(x, memory)
        
        # Global sync every K chunks
        if global_sync and chunk_idx % sync_cadence == 0 and chunk_idx > 0:
            with torch.no_grad():
                pos = torch.arange(0, x.size(1), device=device).long()
                token_repr = model.tok_emb(x) + model.pos_emb(pos)
            _, new_memory = global_sync(token_repr, new_memory)
        
        # Compute loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if global_sync:
            torch.nn.utils.clip_grad_norm_(global_sync.parameters(), max_grad_norm)
        optimizer.step()
        
        # Detach memory for truncated BPTT
        memory = new_memory.detach()
        
        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()
        num_chunks += 1
    
    return {
        'loss': total_loss / max(num_chunks, 1),
        'accuracy': total_correct / max(total_tokens, 1),
        'num_chunks': num_chunks
    }


# ============================================================================
# SECTION 5: GENERATION UTILITIES
# Autoregressive text generation with memory
# ============================================================================

def generate_with_rsm(
    model: HybridTransformer,
    prompt_tokens: List[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> List[int]:
    """
    Generate tokens autoregressively with RSM.
    
    Args:
        model: Trained HybridTransformer
        prompt_tokens: Initial token IDs
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more confident)
        top_k: Keep only top k tokens (None = disabled)
        top_p: Nucleus sampling threshold (None = disabled)
        eos_token_id: Stop if this token is generated
        device: Device to run on
    
    Returns:
        generated: List of token IDs (including prompt)
    """
    model.eval()
    
    generated = prompt_tokens.copy()
    chunk_size = model.window_size - 1
    
    with torch.no_grad():
        # Initialize memory
        memory = model.memory_init.expand(1, -1, -1)
        
        for _ in range(max_new_tokens):
            # Take last chunk_size tokens
            context = generated[-chunk_size:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            
            # Forward pass
            logits, memory = model(x, memory)
            
            # Get next token logits
            next_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                topk_values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_values[-1]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            # Stop on EOS
            if eos_token_id is not None and next_token == eos_token_id:
                break
    
    return generated


# ============================================================================
# SECTION 6: DATASET UTILITIES
# Flexible dataset creation for different tokenization schemes
# ============================================================================

class ChunkedSequenceDataset(Dataset):
    """
    Generic chunked dataset for sequential token data.
    
    Creates overlapping chunks for training with RSM.
    Compatible with any tokenization scheme.
    """
    
    def __init__(self, tokens: List[int], chunk_size: int = 256, 
                 overlap_ratio: float = 0.5):
        """
        Args:
            tokens: List of token IDs
            chunk_size: Size of each chunk
            overlap_ratio: Overlap between consecutive chunks (0.5 = 50% overlap)
        """
        self.tokens = tokens
        self.chunk_size = chunk_size
        self.stride = int(chunk_size * (1 - overlap_ratio))
        
        self.chunks = []
        for i in range(0, len(tokens) - chunk_size, self.stride):
            chunk = tokens[i:i + chunk_size + 1]
            if len(chunk) == chunk_size + 1:
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.chunks.append((x, y))
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]
    
    def __iter__(self):
        return iter(self.chunks)


# ============================================================================
# SECTION 7: HELPER FUNCTIONS
# Checkpointing, metrics, and utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    filepath: str,
    model: HybridTransformer,
    global_sync: Optional[GlobalSyncLayer] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    history: Optional[Dict[str, List[float]]] = None
):
    """
    Save model checkpoint.
    
    Args:
        filepath: Path to save checkpoint
        model: HybridTransformer model
        global_sync: Optional GlobalSyncLayer
        optimizer: Optional optimizer state
        config: Optional config dictionary
        history: Optional training history
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'n_slots': model.n_slots,
            'window_size': model.window_size,
        }
    }
    
    if global_sync is not None:
        checkpoint['global_sync_state_dict'] = global_sync.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if config is not None:
        checkpoint['config'].update(config)
    
    if history is not None:
        checkpoint['history'] = history
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, device: torch.device = torch.device('cpu')) -> Tuple:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded HybridTransformer
        global_sync: Loaded GlobalSyncLayer (if present)
        config: Configuration dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['config']
    
    # Recreate model
    model = HybridTransformer(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        n_slots=config['n_slots'],
        window_size=config['window_size']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate global sync if present
    global_sync = None
    if 'global_sync_state_dict' in checkpoint:
        global_sync = GlobalSyncLayer(
            hidden_size=config['hidden_size'],
            n_slots=config['n_slots']
        ).to(device)
        global_sync.load_state_dict(checkpoint['global_sync_state_dict'])
    
    print(f"✓ Checkpoint loaded from {filepath}")
    return model, global_sync, config


def create_rsm_model(
    vocab_size: int,
    hidden_size: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    num_memory_slots: int = 32,
    chunk_size: int = 256,
    dropout: float = 0.1,
    use_global_sync: bool = True,
    device: torch.device = torch.device('cpu')
) -> Tuple[HybridTransformer, Optional[GlobalSyncLayer], Dict[str, Any]]:
    """
    Factory function to create RSM model with sensible defaults.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_memory_slots: Number of external memory slots
        chunk_size: Context window size
        dropout: Dropout probability
        use_global_sync: Whether to create GlobalSyncLayer
        device: Device to create model on
    
    Returns:
        model: HybridTransformer
        global_sync: GlobalSyncLayer or None
        config: Configuration dictionary
    """
    config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_memory_slots': num_memory_slots,
        'chunk_size': chunk_size,
        'dropout': dropout,
        'use_global_sync': use_global_sync
    }
    
    model = HybridTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_slots=num_memory_slots,
        window_size=chunk_size,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    global_sync = None
    if use_global_sync:
        global_sync = GlobalSyncLayer(hidden_size, num_memory_slots).to(device)
    
    # Print model info
    model_params = count_parameters(model)
    sync_params = count_parameters(global_sync) if global_sync else 0
    total_params = model_params + sync_params
    
    print("=" * 80)
    print("RSM MODEL CREATED")
    print("=" * 80)
    print(f"Vocabulary: {vocab_size:,} tokens")
    print(f"Hidden size: {hidden_size}")
    print(f"Layers: {num_layers}")
    print(f"Attention heads: {num_heads}")
    print(f"Memory slots: {num_memory_slots}")
    print(f"Chunk size: {chunk_size}")
    print(f"Global sync: {'Yes' if use_global_sync else 'No'}")
    print(f"\nParameters:")
    print(f"  Main model: {model_params:,}")
    print(f"  Global sync: {sync_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"  Memory (float32): {total_params * 4 / 1024**2:.1f} MB")
    print("=" * 80)
    
    return model, global_sync, config


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # NNDL Core
    'scaled_dot_attention',
    'PositionalEncoding',
    'MLP',
    'CausalSelfAttention',
    
    # Memory Components
    'CrossAttention',
    'GatedSSM',
    'GlobalSyncLayer',
    
    # Architecture
    'HybridTransformerBlock',
    'HybridTransformer',
    
    # Training
    'train_rsm_epoch',
    
    # Generation
    'generate_with_rsm',
    
    # Dataset
    'ChunkedSequenceDataset',
    
    # Helpers
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint',
    'create_rsm_model',
]
print('✓ hybrid_transformer1 module loaded.')