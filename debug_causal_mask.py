#!/usr/bin/env python3
import torch

# Reproduce the issue
batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 8

query = torch.randn(batch_size, seq_len, num_heads, head_dim)
key = torch.randn(batch_size, seq_len, num_heads, head_dim)

print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")

# Compute attention scores
scores = torch.matmul(query, key.transpose(-2, -1))
print(f"Scores shape: {scores.shape}")

# Create causal mask
mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool))
print(f"Initial mask shape: {mask.shape}")

mask = mask.unsqueeze(0).unsqueeze(2)  
print(f"Expanded mask shape: {mask.shape}")

print(f"Trying to mask scores {scores.shape} with mask {mask.shape}")