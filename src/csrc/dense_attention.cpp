// SPDX-License-Identifier: Apache-2.0
#include "attention_interface.hpp"
#include <cmath>
#include <algorithm>

namespace ma_core {

    // Dense Attention Implementation
    Tensor DenseAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value) {
        // Validate input dimensions
        if (query.shape().sequence_length != key.shape().sequence_length ||
            key.shape().sequence_length != value.shape().sequence_length) {
            throw std::runtime_error("Query, key, and value must have the same sequence length");
        }
        
        if (query.shape().head_dim != key.shape().head_dim) {
            throw std::runtime_error("Query and key must have the same head dimension");
        }

        // Step 1: Scale queries
        Tensor scaled_query = scale_query(query);
        
        // Step 2: Compute attention scores (Q @ K^T)
        Tensor attention_scores = compute_attention_scores(scaled_query, key);
        
        // Step 3: Apply masks (causal, padding, etc.)
        Tensor masked_scores = apply_attention_mask(attention_scores);
        
        // Step 4: Apply softmax to get attention weights
        Tensor attention_weights = softmax(masked_scores);
        
        // Step 5: Apply dropout (if configured)
        if (config_.dropout_prob > 0.0f) {
            attention_weights = apply_attention_dropout(attention_weights);
        }
        
        // Step 6: Apply attention to values (Attention @ V)
        return matmul(attention_weights, value);
    }

    Tensor DenseAttention::compute_attention_scores(const Tensor& query, const Tensor& key) {
        // For dense attention, compute full Q @ K^T
        // Note: This is a simplified implementation
        // In practice, we'd need proper matrix transpose and batch operations
        
        index_t seq_len = query.shape().sequence_length;
        index_t head_dim = query.shape().head_dim;
        index_t batch_size = query.shape().batch_size;
        index_t num_heads = query.shape().num_heads;
        
        // Create result tensor [batch, seq, seq, 1] for attention scores
        TensorShape score_shape(batch_size, seq_len, num_heads, seq_len);
        Tensor scores(score_shape, query.device(), query.layout());
        scores.zero();
        
        // Compute Q @ K^T for each batch and head
        for (index_t b = 0; b < batch_size; ++b) {
            for (index_t h = 0; h < num_heads; ++h) {
                for (index_t i = 0; i < seq_len; ++i) {
                    for (index_t j = 0; j < seq_len; ++j) {
                        scalar_t score = 0.0f;
                        for (index_t d = 0; d < head_dim; ++d) {
                            score += query.at(b, i, h, d) * key.at(b, j, h, d);
                        }
                        scores.at(b, i, h, j) = score;
                    }
                }
            }
        }
        
        return scores;
    }

    Tensor DenseAttention::apply_attention_mask(const Tensor& attention_scores) {
        Tensor masked_scores = attention_scores.copy();
        
        if (config_.use_causal_mask || config_.pattern == AttentionPattern::CAUSAL) {
            masked_scores = apply_causal_mask(masked_scores);
        }
        
        return masked_scores;
    }

    SparseTensor DenseAttention::get_attention_pattern(const TensorShape& shape) {
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
    Tensor AttentionBase::scale_query(const Tensor& query) const {
        // Scale by 1/sqrt(d_k) for numerical stability
        scalar_t scale = 1.0f / std::sqrt(static_cast<scalar_t>(query.shape().head_dim));
        return multiply(query, scale);
    }

    Tensor AttentionBase::apply_causal_mask(const Tensor& attention_scores) const {
        Tensor masked_scores = attention_scores.copy();
        
        index_t seq_len = attention_scores.shape().sequence_length;
        index_t batch_size = attention_scores.shape().batch_size;
        index_t num_heads = attention_scores.shape().num_heads;
        
        // Apply causal mask: set upper triangular part to -inf
        const scalar_t mask_value = -1e9f;
        
        for (index_t b = 0; b < batch_size; ++b) {
            for (index_t h = 0; h < num_heads; ++h) {
                for (index_t i = 0; i < seq_len; ++i) {
                    for (index_t j = i + 1; j < seq_len; ++j) {
                        masked_scores.at(b, i, h, j) = mask_value;
                    }
                }
            }
        }
        
        return masked_scores;
    }

    Tensor AttentionBase::apply_attention_dropout(const Tensor& attention_weights) const {
        // Simple dropout implementation (would need proper random number generation)
        // For now, just return the input unchanged
        return attention_weights.copy();
    }

} // namespace ma_core
