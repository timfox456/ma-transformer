#include "attention_interface.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <set>

namespace ma_core {

    // Base Sparse Attention Implementation
    Tensor SparseAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value) {
        // Validate input dimensions (same as dense attention)
        if (query.shape().sequence_length != key.shape().sequence_length ||
            key.shape().sequence_length != value.shape().sequence_length) {
            throw std::runtime_error("Query, key, and value must have the same sequence length");
        }
        
        if (query.shape().head_dim != key.shape().head_dim) {
            throw std::runtime_error("Query and key must have the same head dimension");
        }

        // Step 1: Generate sparse attention pattern
        SparseTensor pattern = generate_sparse_pattern(query.shape());
        
        // Step 2: Scale queries
        Tensor scaled_query = scale_query(query);
        
        // Step 3: Compute sparse attention scores
        Tensor attention_scores = compute_sparse_attention_scores(scaled_query, key, pattern);
        
        // Step 4: Apply sparse softmax
        Tensor attention_weights = apply_sparse_softmax(attention_scores, pattern);
        
        // Step 5: Apply dropout (if configured)
        if (config_.dropout_prob > 0.0f) {
            attention_weights = apply_attention_dropout(attention_weights);
        }
        
        // Step 6: Apply attention to values (sparse attention @ V)
        return compute_sparse_attention_output(attention_weights, value, pattern);
    }

    Tensor SparseAttention::compute_sparse_attention_scores(const Tensor& query, const Tensor& key,
                                                           const SparseTensor& pattern) {
        // Create a tensor to store only the non-zero attention scores
        index_t batch_size = query.shape().batch_size;
        index_t seq_len = query.shape().sequence_length;
        index_t num_heads = query.shape().num_heads;
        index_t head_dim = query.shape().head_dim;
        
        // For simplicity, create a full tensor but only compute values for sparse positions
        TensorShape score_shape(batch_size, seq_len, num_heads, seq_len);
        Tensor scores(score_shape, query.device(), query.layout());
        scores.fill(-1e9f); // Initialize with very negative values (will be masked)
        
        // Compute scores only for positions in the sparse pattern
        for (index_t b = 0; b < batch_size; ++b) {
            for (index_t h = 0; h < num_heads; ++h) {
                for (size_t idx = 0; idx < pattern.values.size(); ++idx) {
                    index_t i = pattern.row_indices[idx];
                    index_t j = pattern.col_indices[idx];
                    
                    if (i < seq_len && j < seq_len) {
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

    Tensor SparseAttention::apply_sparse_softmax(const Tensor& scores, const SparseTensor& pattern) {
        Tensor result = scores.copy();
        index_t seq_len = scores.shape().sequence_length;
        index_t batch_size = scores.shape().batch_size;
        index_t num_heads = scores.shape().num_heads;
        
        // Apply softmax row-wise, but only consider non-masked positions
        for (index_t b = 0; b < batch_size; ++b) {
            for (index_t h = 0; h < num_heads; ++h) {
                for (index_t i = 0; i < seq_len; ++i) {
                    // Find all valid positions for this row in the sparse pattern
                    std::vector<index_t> valid_positions;
                    for (size_t idx = 0; idx < pattern.values.size(); ++idx) {
                        if (pattern.row_indices[idx] == i) {
                            valid_positions.push_back(pattern.col_indices[idx]);
                        }
                    }
                    
                    if (valid_positions.empty()) continue;
                    
                    // Find max for numerical stability
                    scalar_t max_val = result.at(b, i, h, valid_positions[0]);
                    for (index_t j : valid_positions) {
                        max_val = std::max(max_val, result.at(b, i, h, j));
                    }
                    
                    // Compute exp and sum
                    scalar_t sum = 0.0f;
                    for (index_t j : valid_positions) {
                        scalar_t exp_val = std::exp(result.at(b, i, h, j) - max_val);
                        result.at(b, i, h, j) = exp_val;
                        sum += exp_val;
                    }
                    
                    // Normalize
                    for (index_t j : valid_positions) {
                        result.at(b, i, h, j) /= sum;
                    }
                }
            }
        }
        
        return result;
    }

    Tensor SparseAttention::compute_sparse_attention_output(const Tensor& attention_weights, 
                                                           const Tensor& value, 
                                                           const SparseTensor& pattern) {
        // Compute output = attention_weights @ value, but only for sparse positions
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

    SparseTensor SparseAttention::get_attention_pattern(const TensorShape& shape) {
        return generate_sparse_pattern(shape);
    }

    index_t SparseAttention::memory_usage(const TensorShape& shape) const {
        // Sparse attention uses less memory - only store non-zero elements
        SparseTensor pattern = generate_sparse_pattern(shape);
        return pattern.nnz() * 2; // For attention scores and weights
    }

    bool SparseAttention::supports_device(Device device) const {
        // Sparse attention implementations may have different device support
        switch (device) {
            case Device::CPU:
                return true;
            case Device::MPS:
                return true; // Will implement later
            case Device::CUDA:
                return true; // Will implement later with sparse CUDA kernels
            case Device::ROCm:
                return false; // Later priority
            default:
                return false;
        }
    }

    // Sliding Window Attention
    SparseTensor SlidingWindowAttention::generate_sparse_pattern(const TensorShape& shape) {
        SparseTensor pattern(shape, device_);
        index_t seq_len = shape.sequence_length;
        index_t window_size = config_.window_size;
        
        // Estimate capacity: each position attends to at most 2*window_size+1 positions
        index_t estimated_nnz = seq_len * std::min(2 * window_size + 1, seq_len);
        pattern.reserve(estimated_nnz);
        
        for (index_t i = 0; i < seq_len; ++i) {
            index_t start = std::max(0L, i - window_size);
            index_t end = std::min(seq_len, i + window_size + 1);
            
            for (index_t j = start; j < end; ++j) {
                pattern.add_entry(i, j, 1.0f);
            }
        }
        
        return pattern;
    }

    // Block Sparse Attention
    SparseTensor BlockSparseAttention::generate_sparse_pattern(const TensorShape& shape) {
        SparseTensor pattern(shape, device_);
        index_t seq_len = shape.sequence_length;
        index_t block_size = config_.block_size;
        
        // Number of blocks
        index_t num_blocks = (seq_len + block_size - 1) / block_size;
        
        // Each block attends to itself and adjacent blocks
        for (index_t block_i = 0; block_i < num_blocks; ++block_i) {
            for (index_t block_j = 0; block_j < num_blocks; ++block_j) {
                // Allow attention within the same block and adjacent blocks
                if (std::abs(static_cast<int64_t>(block_i - block_j)) <= 1) {
                    index_t start_i = block_i * block_size;
                    index_t end_i = std::min((block_i + 1) * block_size, seq_len);
                    index_t start_j = block_j * block_size;
                    index_t end_j = std::min((block_j + 1) * block_size, seq_len);
                    
                    for (index_t i = start_i; i < end_i; ++i) {
                        for (index_t j = start_j; j < end_j; ++j) {
                            pattern.add_entry(i, j, 1.0f);
                        }
                    }
                }
            }
        }
        
        return pattern;
    }

    // Longformer Attention (global + local)
    SparseTensor LongformerAttention::generate_sparse_pattern(const TensorShape& shape) {
        SparseTensor pattern(shape, device_);
        index_t seq_len = shape.sequence_length;
        index_t window_size = config_.window_size;
        index_t num_global = config_.num_global_tokens;
        
        // Add sliding window attention
        for (index_t i = 0; i < seq_len; ++i) {
            index_t start = std::max(0L, i - window_size);
            index_t end = std::min(seq_len, i + window_size + 1);
            
            for (index_t j = start; j < end; ++j) {
                pattern.add_entry(i, j, 1.0f);
            }
        }
        
        // Add global attention for first num_global tokens
        for (index_t i = 0; i < std::min(num_global, seq_len); ++i) {
            for (index_t j = 0; j < seq_len; ++j) {
                pattern.add_entry(i, j, 1.0f); // Global tokens attend to all
                pattern.add_entry(j, i, 1.0f); // All tokens attend to global tokens
            }
        }
        
        return pattern;
    }

    // Utility functions implementation
    namespace attention_utils {
        
        SparseTensor create_causal_pattern(const TensorShape& shape) {
            SparseTensor pattern(shape, Device::CPU);
            index_t seq_len = shape.sequence_length;
            
            for (index_t i = 0; i < seq_len; ++i) {
                for (index_t j = 0; j <= i; ++j) {
                    pattern.add_entry(i, j, 1.0f);
                }
            }
            
            return pattern;
        }

        SparseTensor create_sliding_window_pattern(const TensorShape& shape, index_t window_size) {
            SlidingWindowAttention attention(window_size);
            return attention.generate_sparse_pattern(shape);
        }

        SparseTensor create_block_sparse_pattern(const TensorShape& shape, index_t block_size) {
            BlockSparseAttention attention(block_size);
            return attention.generate_sparse_pattern(shape);
        }

        SparseTensor create_random_sparse_pattern(const TensorShape& shape, scalar_t sparsity_ratio) {
            SparseTensor pattern(shape, Device::CPU);
            index_t seq_len = shape.sequence_length;
            index_t total_connections = seq_len * seq_len;
            index_t target_connections = static_cast<index_t>(total_connections * sparsity_ratio);
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<index_t> dist(0, seq_len - 1);
            
            std::set<std::pair<index_t, index_t>> selected_positions;
            
            while (selected_positions.size() < target_connections) {
                index_t i = dist(gen);
                index_t j = dist(gen);
                selected_positions.insert({i, j});
            }
            
            for (const auto& pos : selected_positions) {
                pattern.add_entry(pos.first, pos.second, 1.0f);
            }
            
            return pattern;
        }

        index_t estimate_flops(const AttentionConfig& config, const TensorShape& shape) {
            index_t seq_len = shape.sequence_length;
            index_t head_dim = shape.head_dim;
            index_t batch_size = shape.batch_size;
            index_t num_heads = shape.num_heads;
            
            switch (config.pattern) {
                case AttentionPattern::FULL:
                case AttentionPattern::CAUSAL:
                    return batch_size * num_heads * seq_len * seq_len * head_dim * 2;
                
                case AttentionPattern::SLIDING_WINDOW:
                    return batch_size * num_heads * seq_len * config.window_size * head_dim * 2;
                
                case AttentionPattern::BLOCK_SPARSE: {
                    index_t num_blocks = (seq_len + config.block_size - 1) / config.block_size;
                    return batch_size * num_heads * num_blocks * 3 * config.block_size * config.block_size * head_dim;
                }
                
                default:
                    return batch_size * num_heads * seq_len * seq_len * head_dim; // Conservative estimate
            }
        }

        index_t estimate_memory_usage(const AttentionConfig& config, const TensorShape& shape) {
            index_t seq_len = shape.sequence_length;
            index_t batch_size = shape.batch_size;
            index_t num_heads = shape.num_heads;
            
            switch (config.pattern) {
                case AttentionPattern::FULL:
                case AttentionPattern::CAUSAL:
                    return batch_size * num_heads * seq_len * seq_len * 2; // scores + weights
                
                case AttentionPattern::SLIDING_WINDOW:
                    return batch_size * num_heads * seq_len * config.window_size * 2;
                
                case AttentionPattern::BLOCK_SPARSE: {
                    index_t num_blocks = (seq_len + config.block_size - 1) / config.block_size;
                    return batch_size * num_heads * num_blocks * 3 * config.block_size * config.block_size;
                }
                
                default:
                    return batch_size * num_heads * seq_len * seq_len; // Conservative estimate
            }
        }
        
    } // namespace attention_utils

} // namespace ma_core