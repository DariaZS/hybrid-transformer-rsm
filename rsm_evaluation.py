"""
RSM Evaluation Functions
========================

Helper functions for evaluating the Rolling State Memory transformer.
Implements evaluation plan from project proposal (sections 8 & 9).

Author: Daria Strait, Peter Vail Driscoll
Course: Deep Learning (Fall 2025)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from hybrid_transformer1 import (GlobalSyncLayer, HybridTransformer,
                                 count_parameters, create_rsm_model)

# ==============================================================================
# 1. RETRIEVAL AT DISTANCE
# ==============================================================================

class FactRetrievalDataset:
    """
    Creates synthetic fact retrieval tasks for testing memory.
    
    The idea is to put a fact early in the sequence, then add lots of filler,
    then ask about the fact later. This tests if the model can remember
    things from way back.
    
    Format: "fact: A=5 ... [lots of random tokens] ... query: A=?"
    """
    
    def __init__(self, vocab_size=1000, num_facts=100):
        self.vocab_size = vocab_size
        self.num_facts = num_facts
        
        # Generate random entity-value pairs as our "facts"
        # In a real scenario these would be like "Paris is capital of France"
        # but we just use random token IDs for simplicity
        self.facts = []
        for _ in range(num_facts):
            fact = {
                'entity': torch.randint(10, 100, (1,)).item(),
                'value': torch.randint(100, 200, (1,)).item(),
                'query_token': torch.randint(200, 300, (1,)).item()
            }
            self.facts.append(fact)
    
    def generate_sequence(self, fact_idx, distance, seq_len=256):
        """
        Make a sequence with fact at start, query at distance.
        
        Args:
            fact_idx: which fact to use
            distance: tokens between fact and query
            seq_len: total sequence length (default 256 to match typical model training)
            
        Returns:
            tokens, answer, positions dict
        """
        fact = self.facts[fact_idx]
        sequence = []
        
        # Start with the fact
        sequence.extend([fact['entity'], fact['value']])
        fact_pos = 0
        
        # Add filler tokens - just random stuff
        num_filler = min(distance - 4, seq_len - 10)  # leave room for fact and query
        if num_filler > 0:
            filler = torch.randint(300, self.vocab_size, (num_filler,)).tolist()
            sequence.extend(filler)
        
        # Add query at the end
        query_pos = len(sequence)
        sequence.extend([fact['query_token'], fact['entity'], 999])  # 999 = "what is it?"
        
        # Pad to seq_len
        if len(sequence) < seq_len:
            sequence.extend([0] * (seq_len - len(sequence)))
        else:
            sequence = sequence[:seq_len]
        
        return torch.tensor(sequence), fact['value'], {'fact_pos': fact_pos, 'query_pos': query_pos}


def evaluate_retrieval_at_distance(model, distances=[128, 256, 512, 1024, 2048], 
                                   num_samples=50, device='cpu', max_seq_len=256):
    """
    Test if model can recall facts from different distances.
    
    This is one of the key metrics from our proposal - does the model
    remember things from way back in the context?
    
    Args:
        model: The model to evaluate
        distances: List of distances to test
        num_samples: Number of samples per distance
        device: Device to run on
        max_seq_len: Maximum sequence length (default 256 to match typical training)
    """
    model.eval()
    dataset = FactRetrievalDataset(vocab_size=model.vocab_size, num_facts=100)
    results = {}
    
    for distance in distances:
        correct = 0
        total = 0
        
        print(f"Testing distance={distance}...")
        
        # Skip distances that are too long for the model
        if distance >= max_seq_len - 10:
            print(f"  Skipping - distance {distance} too large for max_seq_len={max_seq_len}")
            results[distance] = {'accuracy': 0.0, 'skipped': True}
            continue
        
        with torch.no_grad():
            for i in range(num_samples):
                fact_idx = i % len(dataset.facts)
                sequence, target, positions = dataset.generate_sequence(fact_idx, distance, seq_len=max_seq_len)
                
                # Run through model
                sequence = sequence.unsqueeze(0).to(device)
                logits, _ = model(sequence)
                
                # Check what it predicts at the query position
                query_pos = positions['query_pos'] + 2  # after the "?"
                if query_pos < logits.size(1):
                    pred = logits[0, query_pos].argmax().item()
                    if pred == target:
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        results[distance] = {'accuracy': accuracy, 'correct': correct, 'total': total}
        print(f"  Distance {distance}: {accuracy:.3f} accuracy ({correct}/{total})")
    
    return results


# ==============================================================================
# 2. CHAIN-OF-THOUGHT DEPTH
# ==============================================================================

class ChainOfThoughtTask:
    """
    Multi-step reasoning tasks to test how deep the model can "think".
    
    Example (depth 3):
        step1: 5 + 3 = 8
        step2: 8 * 2 = 16  
        step3: 16 - 4 = 12
        answer: 12
        
    Tests if model can follow multiple steps of reasoning.
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.ops = ['+', '-', '*']
        
        # Token mappings - need to encode math operations as tokens
        self.num_offset = 10  # numbers start at token 10
        self.op_tokens = {'+': 1, '-': 2, '*': 3, '=': 4}
        self.step_token = 5
        self.answer_token = 6
    
    def generate_task(self, depth):
        """
        Generate a reasoning task with 'depth' steps.
        
        Each step does one math operation, building on previous result.
        """
        sequence = []
        current_value = torch.randint(1, 10, (1,)).item()
        
        # Generate chain of operations
        for step in range(depth):
            op = self.ops[torch.randint(0, len(self.ops), (1,)).item()]
            operand = torch.randint(1, 5, (1,)).item()
            
            # Do the math
            if op == '+':
                result = current_value + operand
            elif op == '-':
                result = max(1, current_value - operand)  # keep it positive
            else:  # '*'
                result = current_value * operand
            
            # Encode as: [STEP, num1, OP, num2, =, result]
            sequence.extend([
                self.step_token,
                self.num_offset + current_value,
                self.op_tokens[op],
                self.num_offset + operand,
                self.op_tokens['='],
                self.num_offset + result
            ])
            
            current_value = result
        
        # Add final query
        sequence.extend([self.answer_token, 999])
        
        return torch.tensor(sequence), current_value


def evaluate_cot_depth(model, max_depth=10, num_samples=30, device='cpu'):
    """
    Test how many reasoning steps the model can handle.
    
    We expect accuracy to drop as depth increases - question is how fast?
    """
    model.eval()
    task_gen = ChainOfThoughtTask(vocab_size=model.vocab_size)
    results = {'depth': [], 'accuracy': []}
    
    for depth in range(1, max_depth + 1):
        correct = 0
        total = 0
        
        print(f"Testing depth={depth}...")
        
        with torch.no_grad():
            for _ in range(num_samples):
                sequence, answer = task_gen.generate_task(depth)
                
                sequence = sequence.unsqueeze(0).to(device)
                logits, _ = model(sequence)
                
                # Check last position
                pred = logits[0, -1].argmax().item()
                expected_token = task_gen.num_offset + answer
                
                if pred == expected_token:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        results['depth'].append(depth)
        results['accuracy'].append(accuracy)
        print(f"  Depth {depth}: {accuracy:.3f} accuracy")
    
    return results


# ==============================================================================
# 3. ABLATION STUDIES
# ==============================================================================

def ablation_no_sync(vocab_size, distances=[512, 1024], num_samples=30, device='cpu'):
    """
    Ablation 1: Remove global sync layer.
    
    From proposal: "expect drop at multiples of K"
    The idea is that without sync, info doesn't propagate through memory well.
    """
    print("\n=== Ablation 1: No Global Sync ===")
    
    # Create model without sync
    model, _, config = create_rsm_model(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=4,
        num_memory_slots=32,
        chunk_size=256,
        use_global_sync=False,  # THIS IS THE ABLATION
        device=device
    )
    
    # Test it
    results = evaluate_retrieval_at_distance(
        model, 
        distances=distances, 
        num_samples=num_samples,
        device=device
    )
    
    return results


def ablation_freeze_ssm(model, distances=[512, 1024], num_samples=30, device='cpu'):
    """
    Ablation 2: Freeze the SSM memory writes.
    
    This prevents memory from updating, so we expect it to fail on long sequences.
    """
    print("\n=== Ablation 2: Freeze SSM Writes ===")
    
    # Freeze memory_write params in all blocks
    frozen_params = 0
    for block in model.blocks:
        for param in block.memory_write.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"Froze {frozen_params:,} SSM parameters")
    
    results = evaluate_retrieval_at_distance(
        model,
        distances=distances,
        num_samples=num_samples,
        device=device
    )
    
    # Unfreeze afterwards
    for block in model.blocks:
        for param in block.memory_write.parameters():
            param.requires_grad = True
    
    return results


def ablation_vary_architecture(vocab_size, 
                               memory_slots_list=[16, 32, 64],
                               chunk_sizes_list=[128, 256, 512],
                               device='cpu'):
    """
    Ablation 3: Try different memory sizes and chunk sizes.
    
    This gives us the capacity vs cost tradeoff curves.
    Bigger memory = more capacity but more parameters.
    """
    print("\n=== Ablation 3: Architecture Variations ===")
    
    results = {}
    
    for n_slots in memory_slots_list:
        for chunk_size in chunk_sizes_list:
            config_name = f"m{n_slots}_c{chunk_size}"
            print(f"\nTesting {config_name}...")
            
            model, sync, config = create_rsm_model(
                vocab_size=vocab_size,
                hidden_size=128,
                num_layers=4,
                num_memory_slots=n_slots,
                chunk_size=chunk_size,
                device=device
            )
            
            # Count params (this is the "cost")
            params = count_parameters(model)
            
            # Test on one distance (for speed)
            acc_results = evaluate_retrieval_at_distance(
                model,
                distances=[512],
                num_samples=20,
                device=device
            )
            
            results[config_name] = {
                'parameters': params,
                'accuracy': acc_results[512],
                'memory_slots': n_slots,
                'chunk_size': chunk_size
            }
            
            print(f"  Params: {params:,}, Accuracy: {acc_results[512]:.3f}")
            
            # Clean up GPU memory
            del model, sync
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results


# ==============================================================================
# 4. MEMORY TELEMETRY
# ==============================================================================

class MemoryProbe(nn.Module):
    """
    Linear probe to test what's stored in memory slots.
    
    The question: can we decode what entities/concepts are in each slot?
    If yes, memory is storing useful info. If no, it's just noise.
    """
    
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.probe = nn.Linear(hidden_size, num_classes)
    
    def forward(self, memory_slots):
        """memory_slots: (batch, n_slots, hidden_size)"""
        return self.probe(memory_slots)


def train_memory_probe(model, probe, train_data, num_epochs=10, device='cpu'):
    """
    Train a linear probe on frozen model memory.
    
    This tests "linear decodability" from the proposal.
    We freeze the model and just train a simple linear layer to predict
    what entity is being tracked from the memory state.
    """
    model.eval()  # freeze the main model
    probe.train()
    
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for sequence, entity_label in train_data:
            sequence = sequence.unsqueeze(0).to(device)
            entity_label = entity_label.to(device)
            
            # Get memory from model (no grad)
            with torch.no_grad():
                _, memory = model(sequence)
            
            # Train probe to predict entity from memory
            logits = probe(memory).mean(dim=1)  # average over slots
            loss = F.cross_entropy(logits, entity_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == entity_label).sum().item()
            total += entity_label.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.3f}")
    
    return accuracy


def analyze_memory_retention(model, sequence, device='cpu'):
    """
    Track how memory changes over time - "retention half-lives".
    
    We process a long sequence and save memory at each chunk.
    Then compute similarity between consecutive chunks to see
    how quickly info decays.
    """
    model.eval()
    sequence = sequence.unsqueeze(0).to(device)
    
    memory_history = []
    
    with torch.no_grad():
        chunk_size = model.window_size
        memory = None
        
        # Process in chunks and save memory each time
        for i in range(0, sequence.size(1), chunk_size):
            chunk = sequence[:, i:i+chunk_size]
            if chunk.size(1) == 0:
                break
            
            _, memory = model(chunk, memory)
            memory_history.append(memory.cpu().numpy())
    
    # Compute retention: how similar is memory from t to t+1?
    retention = {}
    n_slots = model.n_slots
    
    for slot_idx in range(n_slots):
        similarities = []
        
        for t in range(len(memory_history) - 1):
            vec_t = memory_history[t][0, slot_idx, :]
            vec_t1 = memory_history[t+1][0, slot_idx, :]
            
            # Cosine similarity
            norm_t = np.linalg.norm(vec_t)
            norm_t1 = np.linalg.norm(vec_t1)
            if norm_t > 0 and norm_t1 > 0:
                sim = np.dot(vec_t, vec_t1) / (norm_t * norm_t1)
            else:
                sim = 0.0
            
            similarities.append(sim)
        
        retention[f'slot_{slot_idx}'] = np.array(similarities)
    
    return retention


# ==============================================================================
# 5. UTILITY FUNCTIONS
# ==============================================================================

def run_full_evaluation(model, vocab_size, device='cpu'):
    """
    Run all evaluation metrics at once.
    
    This is the main function to call - does everything and returns results.
    """
    print("=" * 70)
    print("RUNNING FULL RSM EVALUATION SUITE")
    print("=" * 70)
    
    results = {}
    
    # 1. Retrieval at distance
    print("\n[1/4] Testing retrieval at distance...")
    results['retrieval'] = evaluate_retrieval_at_distance(
        model, 
        distances=[128, 256, 512, 1024, 2048],
        num_samples=50,
        device=device
    )
    
    # 2. CoT depth
    print("\n[2/4] Testing chain-of-thought depth...")
    results['cot'] = evaluate_cot_depth(
        model,
        max_depth=10,
        num_samples=30,
        device=device
    )
    
    # 3. Ablations
    print("\n[3/4] Running ablation studies...")
    results['ablation_no_sync'] = ablation_no_sync(vocab_size, device=device)
    results['ablation_frozen_ssm'] = ablation_freeze_ssm(model, device=device)
    results['ablation_architecture'] = ablation_vary_architecture(vocab_size, device=device)
    
    # 4. Memory retention
    print("\n[4/4] Analyzing memory retention...")
    test_seq = torch.randint(0, vocab_size, (2048,))
    results['retention'] = analyze_memory_retention(model, test_seq, device)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return results
