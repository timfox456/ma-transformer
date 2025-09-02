// SPDX-License-Identifier: Apache-2.0
#include "ma_core.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ma_core {
    
    // Legacy function (updated to int64_t to fix overflow issues)
    int64_t add(int64_t a, int64_t b) {
        return a + b;
    }
    
    // Simple tensor creation for testing
    Tensor create_tensor(const std::vector<int64_t>& shape, const std::string& device) {
        if (shape.size() != 4) {
            throw std::runtime_error("Shape must have 4 dimensions: [batch, seq, heads, dim]");
        }
        TensorShape tensor_shape(shape[0], shape[1], shape[3], shape[2]);  // (batch, seq, head_dim, num_heads)
        return Tensor(tensor_shape, Device::CPU); // Only CPU for now
    }
    
    Tensor create_random_tensor(const std::vector<int64_t>& shape, 
                               float mean, float std,
                               const std::string& device) {
        if (shape.size() != 4) {
            throw std::runtime_error("Shape must have 4 dimensions: [batch, seq, heads, dim]");
        }
        TensorShape tensor_shape(shape[0], shape[1], shape[3], shape[2]);  // (batch, seq, head_dim, num_heads)
        return random_normal(tensor_shape, mean, std, Device::CPU);
    }
    
    // Simple pattern creation for testing
    SparseTensor create_sliding_window_pattern(int64_t seq_len, int64_t window_size) {
        TensorShape shape(1, seq_len, 1, seq_len);
        SparseTensor pattern(shape, Device::CPU);
        
        for (int64_t i = 0; i < seq_len; ++i) {
            int64_t start = std::max(static_cast<int64_t>(0), i - window_size);
            int64_t end = std::min(seq_len, i + window_size + 1);
            
            for (int64_t j = start; j < end; ++j) {
                pattern.add_entry(i, j, 1.0f);
            }
        }
        
        return pattern;
    }
    
    // Simple statistics
    std::string get_pattern_statistics(const SparseTensor& pattern) {
        int64_t seq_len = pattern.shape.sequence_length;
        int64_t total_elements = seq_len * seq_len;
        int64_t nonzero_elements = pattern.nnz();
        float sparsity_ratio = 1.0f - static_cast<float>(nonzero_elements) / total_elements;
        
        std::string result = "Pattern Statistics:\n";
        result += "  Total elements: " + std::to_string(total_elements) + "\n";
        result += "  Non-zero elements: " + std::to_string(nonzero_elements) + "\n";
        result += "  Sparsity ratio: " + std::to_string(sparsity_ratio * 100.0f) + "%\n";
        
        return result;
    }
    
    // Dense attention implementation
    Tensor compute_dense_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                  bool use_causal_mask) {
        // Basic validation
        const TensorShape& q_shape = query.shape();
        const TensorShape& k_shape = key.shape();
        const TensorShape& v_shape = value.shape();
        
        if (q_shape.sequence_length != k_shape.sequence_length || 
            k_shape.sequence_length != v_shape.sequence_length) {
            throw std::runtime_error("Query, key, and value must have same sequence length");
        }
        
        if (q_shape.head_dim != k_shape.head_dim) {
            throw std::runtime_error("Query and key must have same head dimension");
        }
        
        int64_t batch_size = q_shape.batch_size;
        int64_t seq_len = q_shape.sequence_length;
        int64_t num_heads = q_shape.num_heads;
        int64_t head_dim = q_shape.head_dim;
        int64_t value_dim = v_shape.head_dim;
        
        // Create output tensor
        TensorShape output_shape(batch_size, seq_len, value_dim, num_heads);
        Tensor output(output_shape, query.device(), query.layout());
        output.zero();
        
        // Scaling factor for numerical stability
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        // Process each batch and head separately
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t h = 0; h < num_heads; ++h) {
                
                // Step 1: Compute attention scores (Q @ K^T)
                std::vector<std::vector<float>> scores(seq_len, std::vector<float>(seq_len, 0.0f));
                
                for (int64_t i = 0; i < seq_len; ++i) {
                    for (int64_t j = 0; j < seq_len; ++j) {
                        float score = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            score += query.at(b, i, h, d) * key.at(b, j, h, d);
                        }
                        scores[i][j] = score * scale;
                    }
                }
                
                // Step 2: Apply causal mask if requested
                if (use_causal_mask) {
                    for (int64_t i = 0; i < seq_len; ++i) {
                        for (int64_t j = i + 1; j < seq_len; ++j) {
                            scores[i][j] = -1e9f;  // Very negative number
                        }
                    }
                }
                
                // Step 3: Apply softmax row-wise
                for (int64_t i = 0; i < seq_len; ++i) {
                    // Find max for numerical stability
                    float max_score = scores[i][0];
                    for (int64_t j = 1; j < seq_len; ++j) {
                        max_score = std::max(max_score, scores[i][j]);
                    }
                    
                    // Compute exponentials and sum
                    float sum_exp = 0.0f;
                    for (int64_t j = 0; j < seq_len; ++j) {
                        scores[i][j] = std::exp(scores[i][j] - max_score);
                        sum_exp += scores[i][j];
                    }
                    
                    // Normalize
                    for (int64_t j = 0; j < seq_len; ++j) {
                        scores[i][j] /= sum_exp;
                    }
                }
                
                // Step 4: Apply attention to values (Attention @ V)
                for (int64_t i = 0; i < seq_len; ++i) {
                    for (int64_t d = 0; d < value_dim; ++d) {
                        float weighted_sum = 0.0f;
                        for (int64_t j = 0; j < seq_len; ++j) {
                            weighted_sum += scores[i][j] * value.at(b, j, h, d);
                        }
                        output.at(b, i, h, d) = weighted_sum;
                    }
                }
            }
        }
        
        return output;
    }
    
    // Sparse attention implementation with sliding window
    Tensor compute_sparse_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                   int64_t window_size) {
        // Basic validation (same as dense)
        const TensorShape& q_shape = query.shape();
        const TensorShape& k_shape = key.shape();
        const TensorShape& v_shape = value.shape();
        
        if (q_shape.sequence_length != k_shape.sequence_length || 
            k_shape.sequence_length != v_shape.sequence_length) {
            throw std::runtime_error("Query, key, and value must have same sequence length");
        }
        
        if (q_shape.head_dim != k_shape.head_dim) {
            throw std::runtime_error("Query and key must have same head dimension");
        }
        
        int64_t batch_size = q_shape.batch_size;
        int64_t seq_len = q_shape.sequence_length;
        int64_t num_heads = q_shape.num_heads;
        int64_t head_dim = q_shape.head_dim;
        int64_t value_dim = v_shape.head_dim;
        
        // Create output tensor
        TensorShape output_shape(batch_size, seq_len, value_dim, num_heads);
        Tensor output(output_shape, query.device(), query.layout());
        output.zero();
        
        // Scaling factor
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        // Process each batch and head separately
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t h = 0; h < num_heads; ++h) {
                
                // For each query position
                for (int64_t i = 0; i < seq_len; ++i) {
                    // Determine the sliding window range
                    int64_t start_j = std::max(static_cast<int64_t>(0), i - window_size);
                    int64_t end_j = std::min(seq_len, i + window_size + 1);
                    
                    // Step 1: Compute attention scores for window
                    std::vector<float> scores(end_j - start_j);
                    for (int64_t j_idx = 0; j_idx < static_cast<int64_t>(scores.size()); ++j_idx) {
                        int64_t j = start_j + j_idx;
                        float score = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            score += query.at(b, i, h, d) * key.at(b, j, h, d);
                        }
                        scores[j_idx] = score * scale;
                    }
                    
                    // Step 2: Apply softmax over the window
                    if (!scores.empty()) {
                        // Find max for numerical stability
                        float max_score = scores[0];
                        for (size_t j_idx = 1; j_idx < scores.size(); ++j_idx) {
                            max_score = std::max(max_score, scores[j_idx]);
                        }
                        
                        // Compute exponentials and sum
                        float sum_exp = 0.0f;
                        for (size_t j_idx = 0; j_idx < scores.size(); ++j_idx) {
                            scores[j_idx] = std::exp(scores[j_idx] - max_score);
                            sum_exp += scores[j_idx];
                        }
                        
                        // Normalize
                        for (size_t j_idx = 0; j_idx < scores.size(); ++j_idx) {
                            scores[j_idx] /= sum_exp;
                        }
                        
                        // Step 3: Apply attention to values
                        for (int64_t d = 0; d < value_dim; ++d) {
                            float weighted_sum = 0.0f;
                            for (size_t j_idx = 0; j_idx < scores.size(); ++j_idx) {
                                int64_t j = start_j + j_idx;
                                weighted_sum += scores[j_idx] * value.at(b, j, h, d);
                            }
                            output.at(b, i, h, d) = weighted_sum;
                        }
                    }
                }
            }
        }
        
        return output;
    }
    
} // namespace ma_core
