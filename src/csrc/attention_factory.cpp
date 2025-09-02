// SPDX-License-Identifier: Apache-2.0
#include "attention_interface.hpp"
#include <memory>
#include <stdexcept>

namespace ma_core {

    // Factory function implementation
    std::unique_ptr<AttentionBase> create_attention(const AttentionConfig& config, Device device) {
        switch (config.pattern) {
            case AttentionPattern::FULL:
            case AttentionPattern::CAUSAL:
                return std::make_unique<DenseAttention>(config, device);
            
            case AttentionPattern::SLIDING_WINDOW:
                return std::make_unique<SlidingWindowAttention>(config.window_size, device);
            
            case AttentionPattern::BLOCK_SPARSE:
                return std::make_unique<BlockSparseAttention>(config.block_size, device);
            
            case AttentionPattern::LONGFORMER:
                return std::make_unique<LongformerAttention>(config.window_size, config.num_global_tokens, device);
            
            // TODO: Implement additional sparse patterns
            case AttentionPattern::FIXED_PATTERN:
            case AttentionPattern::RANDOM_SPARSE:
            case AttentionPattern::BIG_BIRD:
                throw std::runtime_error("Attention pattern not yet implemented");
            
            default:
                throw std::runtime_error("Unknown attention pattern");
        }
    }

    // Missing function implementations that we need
    namespace {
        // These are helper functions that were referenced but not implemented
        
        Tensor compute_sparse_attention_output(const Tensor& attention_weights, 
                                             const Tensor& value, 
                                             const SparseTensor& pattern) {
            // This was referenced in sparse_attention.cpp but not found in the class
            // Moving the implementation here as a standalone function
            index_t batch_size = value.shape().batch_size;
            index_t seq_len = value.shape().sequence_length;
            index_t num_heads = value.shape().num_heads;
            index_t head_dim = value.shape().head_dim;
            
            Tensor output(value.shape(), value.device(), value.layout());
            output.zero();
            
            // Sparse matrix-vector multiplication
            for (index_t b = 0; b < batch_size; ++b) {
                for (index_t h = 0; h < num_heads; ++h) {
                    for (size_t idx = 0; idx < pattern.values.size(); ++idx) {
                        index_t i = pattern.row_indices[idx];
                        index_t j = pattern.col_indices[idx];
                        
                        if (i < seq_len && j < seq_len) {
                            scalar_t weight = attention_weights.at(b, i, h, j);
                            for (index_t d = 0; d < head_dim; ++d) {
                                output.at(b, i, h, d) += weight * value.at(b, j, h, d);
                            }
                        }
                    }
                }
            }
            
            return output;
        }
    }

    // Add the missing method to SparseAttention class by providing it as a standalone function
    // that can be called from the class method
    Tensor SparseAttention::compute_sparse_attention_output(const Tensor& attention_weights, 
                                                           const Tensor& value, 
                                                           const SparseTensor& pattern) {
        return ::ma_core::compute_sparse_attention_output(attention_weights, value, pattern);
    }

    // Device-specific factory functions for future use
    std::unique_ptr<AttentionBase> create_cpu_attention(const AttentionConfig& config) {
        return create_attention(config, Device::CPU);
    }

    std::unique_ptr<AttentionBase> create_mps_attention(const AttentionConfig& config) {
        auto attention = create_attention(config, Device::MPS);
        if (!attention->supports_device(Device::MPS)) {
            throw std::runtime_error("MPS not supported for this attention type");
        }
        return attention;
    }

    std::unique_ptr<AttentionBase> create_cuda_attention(const AttentionConfig& config) {
        auto attention = create_attention(config, Device::CUDA);
        if (!attention->supports_device(Device::CUDA)) {
            throw std::runtime_error("CUDA not supported for this attention type");
        }
        return attention;
    }

    // Utility functions for attention pattern analysis
    namespace attention_utils {
        
        void print_attention_pattern(const SparseTensor& pattern) {
            printf("Sparse Attention Pattern:\n");
            printf("Shape: [%ld, %ld, %ld, %ld]\n", 
                   pattern.shape.batch_size, pattern.shape.sequence_length,
                   pattern.shape.num_heads, pattern.shape.head_dim);
            printf("Non-zero elements: %ld\n", pattern.nnz());
            printf("Sparsity: %.2f%%\n", 
                   100.0 * (1.0 - static_cast<double>(pattern.nnz()) / 
                   (pattern.shape.sequence_length * pattern.shape.sequence_length)));
            
            // Print first few entries
            printf("First 10 entries:\n");
            size_t print_count = std::min(static_cast<size_t>(10), pattern.values.size());
            for (size_t i = 0; i < print_count; ++i) {
                printf("  [%ld, %ld] = %.3f\n", 
                       pattern.row_indices[i], pattern.col_indices[i], pattern.values[i]);
            }
        }

        bool validate_attention_pattern(const SparseTensor& pattern, const TensorShape& shape) {
            // Check that all indices are within bounds
            for (size_t i = 0; i < pattern.values.size(); ++i) {
                if (pattern.row_indices[i] >= shape.sequence_length ||
                    pattern.col_indices[i] >= shape.sequence_length ||
                    pattern.row_indices[i] < 0 ||
                    pattern.col_indices[i] < 0) {
                    return false;
                }
            }
            
            // Check that arrays have consistent sizes
            if (pattern.row_indices.size() != pattern.col_indices.size() ||
                pattern.col_indices.size() != pattern.values.size()) {
                return false;
            }
            
            return true;
        }

        // Additional utility: convert dense attention matrix to sparse pattern
        SparseTensor dense_to_sparse(const Tensor& dense_attention, scalar_t threshold = 1e-6f) {
            SparseTensor pattern(dense_attention.shape(), dense_attention.device());
            
            index_t seq_len = dense_attention.shape().sequence_length;
            
            for (index_t i = 0; i < seq_len; ++i) {
                for (index_t j = 0; j < seq_len; ++j) {
                    scalar_t value = dense_attention.at(0, i, 0, j); // Assume single batch/head for simplicity
                    if (std::abs(value) > threshold) {
                        pattern.add_entry(i, j, value);
                    }
                }
            }
            
            return pattern;
        }

        // Utility: compute attention pattern statistics
        struct PatternStats {
            index_t total_elements;
            index_t nonzero_elements;
            scalar_t sparsity_ratio;
            scalar_t avg_connections_per_token;
            index_t max_connections_per_token;
            index_t min_connections_per_token;
        };

        PatternStats compute_pattern_statistics(const SparseTensor& pattern) {
            PatternStats stats;
            index_t seq_len = pattern.shape.sequence_length;
            
            stats.total_elements = seq_len * seq_len;
            stats.nonzero_elements = pattern.nnz();
            stats.sparsity_ratio = 1.0f - static_cast<scalar_t>(stats.nonzero_elements) / stats.total_elements;
            
            // Count connections per token
            std::vector<index_t> connections_per_token(seq_len, 0);
            for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
                connections_per_token[pattern.row_indices[i]]++;
            }
            
            stats.max_connections_per_token = *std::max_element(connections_per_token.begin(), connections_per_token.end());
            stats.min_connections_per_token = *std::min_element(connections_per_token.begin(), connections_per_token.end());
            
            index_t total_connections = 0;
            for (index_t count : connections_per_token) {
                total_connections += count;
            }
            stats.avg_connections_per_token = static_cast<scalar_t>(total_connections) / seq_len;
            
            return stats;
        }
        
    } // namespace attention_utils

} // namespace ma_core
