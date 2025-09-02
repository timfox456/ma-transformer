// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention_types.hpp"
#include "tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ma_core {
    /**
     * Legacy arithmetic operations (for backward compatibility)
     */
    int64_t add(int64_t a, int64_t b);
    
    // Basic tensor creation utilities for Python interface
    Tensor create_tensor(const std::vector<int64_t>& shape, const std::string& device = "cpu");
    Tensor create_random_tensor(const std::vector<int64_t>& shape, 
                               float mean = 0.0f, float std = 1.0f,
                               const std::string& device = "cpu");
    
    // Basic sparse pattern creation for testing
    SparseTensor create_sliding_window_pattern(int64_t seq_len, int64_t window_size);
    
    // Basic pattern statistics
    std::string get_pattern_statistics(const SparseTensor& pattern);
    
    // Attention computation
    Tensor compute_dense_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                  bool use_causal_mask = false);
    
    Tensor compute_sparse_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                   int64_t window_size = 64);
    
} // namespace ma_core
