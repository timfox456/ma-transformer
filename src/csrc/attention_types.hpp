// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>

namespace ma_core {
    
    // Core data types
    using scalar_t = float;  // Use float for tensor operations (can template later)
    using index_t = int64_t; // For indices and sizes

    /**
     * Device enumeration for cross-platform support
     */
    enum class Device {
        CPU,     // CPU fallback (always available)
        MPS,     // Metal Performance Shaders (macOS)
        CUDA,    // NVIDIA CUDA
        ROCm     // AMD ROCm
    };

    /**
     * Attention pattern types
     */
    enum class AttentionPattern {
        // Dense patterns
        FULL,           // Standard full attention O(n²)
        CAUSAL,         // Causal (lower triangular) attention
        
        // Sparse patterns
        SLIDING_WINDOW, // Local sliding window
        FIXED_PATTERN,  // Fixed sparse pattern
        RANDOM_SPARSE,  // Random sparse pattern
        BLOCK_SPARSE,   // Block-wise sparse attention
        LONGFORMER,     // Longformer-style (global + local)
        BIG_BIRD        // BigBird-style (global + random + sliding)
    };

    /**
     * Attention configuration parameters
     */
    struct AttentionConfig {
        AttentionPattern pattern;
        index_t window_size = 64;        // For sliding window
        index_t block_size = 64;         // For block sparse
        index_t num_global_tokens = 2;   // For Longformer/BigBird
        index_t num_random_blocks = 3;   // For random sparse patterns
        scalar_t sparsity_ratio = 0.1f;  // For random sparse
        bool use_causal_mask = false;    // Apply causal masking
        scalar_t dropout_prob = 0.0f;    // Attention dropout
        
        // Constructor with defaults
        AttentionConfig(AttentionPattern p = AttentionPattern::FULL) 
            : pattern(p) {}
    };

    /**
     * Tensor shape information
     */
    struct TensorShape {
        index_t batch_size;
        index_t sequence_length;
        index_t head_dim;
        index_t num_heads;
        
        TensorShape(index_t b, index_t s, index_t h, index_t d) 
            : batch_size(b), sequence_length(s), head_dim(h), num_heads(d) {}
        
        index_t total_elements() const {
            return batch_size * sequence_length * head_dim * num_heads;
        }
    };

    /**
     * Memory layout for attention computations
     */
    enum class MemoryLayout {
        ROW_MAJOR,     // C-style: [batch, seq, heads, dim]
        COL_MAJOR,     // Fortran-style
        NHWD,          // [batch, heads, seq, dim] - common in transformers
        NWHD           // [batch, seq, heads, dim] - PyTorch default
    };

} // namespace ma_core
