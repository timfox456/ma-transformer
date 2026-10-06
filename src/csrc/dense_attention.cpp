// SPDX-License-Identifier: Apache-2.0
#include "attention_interface.hpp"
#include "ma_core.hpp"
#include <cmath>
#include <algorithm>

namespace ma_core {

    // Dense Attention Implementation
    Tensor DenseAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value) {
        validate_inputs(query, key, value);
        bool causal = config_.use_causal_mask || config_.pattern == AttentionPattern::CAUSAL;
        return compute_dense_attention(query, key, value, causal);
    }

    SparseTensor DenseAttention::get_attention_pattern(const TensorShape& shape) const {
        return generate_sparse_pattern(shape);
    }

    SparseTensor DenseAttention::generate_sparse_pattern(const TensorShape& shape) const {
        SparseTensor pattern(shape, device_);
        
        index_t seq_len = shape.sequence_length;
        
        // For dense attention, all positions attend to all positions
        if (config_.pattern == AttentionPattern::FULL) {
            pattern.reserve(seq_len * seq_len);
            for (index_t i = 0; i < seq_len; ++i) {
                for (index_t j = 0; j < seq_len; ++j) {
                    pattern.add_entry(i, j, 1.0f);
                }
            }
        } else if (config_.pattern == AttentionPattern::CAUSAL) {
            // Causal attention: only attend to previous positions
            index_t num_entries = seq_len * (seq_len + 1) / 2;
            pattern.reserve(num_entries);
            for (index_t i = 0; i < seq_len; ++i) {
                for (index_t j = 0; j <= i; ++j) {
                    pattern.add_entry(i, j, 1.0f);
                }
            }
        }
        
        return pattern;
    }

    index_t DenseAttention::memory_usage(const TensorShape& shape) const {
        index_t seq_len = shape.sequence_length;
        index_t batch_size = shape.batch_size;
        index_t num_heads = shape.num_heads;
        
        // Memory for attention scores matrix
        index_t attention_scores_memory = batch_size * num_heads * seq_len * seq_len;
        
        // Memory for attention weights (same size)
        index_t attention_weights_memory = attention_scores_memory;
        
        // Total memory in elements (multiply by sizeof(scalar_t) for bytes)
        return attention_scores_memory + attention_weights_memory;
    }

    bool DenseAttention::supports_device(Device device) const {
        // Dense attention should work on all devices
        switch (device) {
            case Device::CPU:
                return true;
            case Device::MPS:
                return true; // Will implement later
            case Device::CUDA:
                return true; // Will implement later
            case Device::ROCm:
                return true; // Will implement later
            default:
                return false;
        }
    }

    // Helper functions implementation
    void AttentionBase::validate_inputs(const Tensor& query, const Tensor& key, const Tensor& value) const {
        const TensorShape& q = query.shape();
        const TensorShape& k = key.shape();
        const TensorShape& v = value.shape();
        if (q.sequence_length != k.sequence_length || k.sequence_length != v.sequence_length) {
            throw std::runtime_error("Query, key, and value must have the same sequence length");
        }
        if (q.batch_size != k.batch_size || k.batch_size != v.batch_size) {
            throw std::runtime_error("Query, key, and value must have the same batch size");
        }
        if (q.num_heads != k.num_heads || k.num_heads != v.num_heads) {
            throw std::runtime_error("Query, key, and value must have the same number of heads");
        }
        if (q.head_dim != k.head_dim) {
            throw std::runtime_error("Query and key must have the same head dimension");
        }
        if (config_.dropout_prob > 0.0f) {
            throw std::invalid_argument(
                "Attention dropout is not implemented in ma_core; apply dropout in PyTorch instead");
        }
    }

} // namespace ma_core
