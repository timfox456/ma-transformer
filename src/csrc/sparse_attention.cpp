// SPDX-License-Identifier: Apache-2.0
#include "attention_interface.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <set>
#include <limits>
#include <map>

namespace ma_core {

    // Base Sparse Attention Implementation
    Tensor SparseAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value) {
        validate_inputs(query, key, value);
        if (query.layout() != MemoryLayout::NWHD || key.layout() != MemoryLayout::NWHD ||
            value.layout() != MemoryLayout::NWHD) {
            throw std::runtime_error("Sparse attention requires [batch, seq, heads, dim] (NWHD) tensors");
        }

        const index_t batch_size = query.shape().batch_size;
        const index_t seq_len = query.shape().sequence_length;
        const index_t num_heads = query.shape().num_heads;
        const index_t head_dim = query.shape().head_dim;
        const index_t value_dim = value.shape().head_dim;

        RowIndex rows = build_row_index(generate_sparse_pattern(query.shape()), seq_len);

        index_t max_row = 0;
        for (index_t i = 0; i < seq_len; ++i) {
            max_row = std::max(max_row, rows.row_ptr[i + 1] - rows.row_ptr[i]);
        }

        Tensor output(value.shape(), value.device(), value.layout());
        output.zero();

        const scalar_t scale = 1.0f / std::sqrt(static_cast<scalar_t>(head_dim));
        const scalar_t* q_data = query.data();
        const scalar_t* k_data = key.data();
        const scalar_t* v_data = value.data();
        scalar_t* o_data = output.data();

        // Row strides in the NWHD layout: consecutive sequence positions are
        // num_heads * dim elements apart.
        const index_t qk_seq_stride = num_heads * head_dim;
        const index_t v_seq_stride = num_heads * value_dim;

        // Scores for one query row at a time: O(max keys per row) scratch
        // instead of an O(seq_len^2) score matrix.
        std::vector<scalar_t> weights(static_cast<size_t>(max_row));

        for (index_t b = 0; b < batch_size; ++b) {
            for (index_t h = 0; h < num_heads; ++h) {
                const index_t qk_base = b * seq_len * qk_seq_stride + h * head_dim;
                const index_t v_base = b * seq_len * v_seq_stride + h * value_dim;

                for (index_t i = 0; i < seq_len; ++i) {
                    const index_t begin = rows.row_ptr[i];
                    const index_t end = rows.row_ptr[i + 1];
                    if (begin == end) continue;

                    const scalar_t* q_row = q_data + qk_base + i * qk_seq_stride;
                    scalar_t max_score = -std::numeric_limits<scalar_t>::infinity();
                    for (index_t n = begin; n < end; ++n) {
                        const scalar_t* k_row = k_data + qk_base + rows.cols[n] * qk_seq_stride;
                        scalar_t score = 0.0f;
                        for (index_t d = 0; d < head_dim; ++d) {
                            score += q_row[d] * k_row[d];
                        }
                        score *= scale;
                        weights[n - begin] = score;
                        max_score = std::max(max_score, score);
                    }

                    scalar_t sum = 0.0f;
                    for (index_t n = 0; n < end - begin; ++n) {
                        weights[n] = std::exp(weights[n] - max_score);
                        sum += weights[n];
                    }

                    scalar_t* o_row = o_data + v_base + i * v_seq_stride;
                    for (index_t n = begin; n < end; ++n) {
                        const scalar_t w = weights[n - begin] / sum;
                        const scalar_t* v_row = v_data + v_base + rows.cols[n] * v_seq_stride;
                        for (index_t d = 0; d < value_dim; ++d) {
                            o_row[d] += w * v_row[d];
                        }
                    }
                }
            }
        }

        return output;
    }

    SparseAttention::RowIndex SparseAttention::build_row_index(const SparseTensor& pattern, index_t seq_len) {
        RowIndex rows;
        rows.row_ptr.assign(static_cast<size_t>(seq_len) + 1, 0);

        auto in_range = [seq_len](index_t i, index_t j) {
            return i >= 0 && i < seq_len && j >= 0 && j < seq_len;
        };

        // Counting sort of the COO entries by row
        for (size_t n = 0; n < pattern.row_indices.size(); ++n) {
            if (in_range(pattern.row_indices[n], pattern.col_indices[n])) {
                ++rows.row_ptr[pattern.row_indices[n] + 1];
            }
        }
        for (index_t i = 0; i < seq_len; ++i) {
            rows.row_ptr[i + 1] += rows.row_ptr[i];
        }
        rows.cols.resize(static_cast<size_t>(rows.row_ptr[seq_len]));
        std::vector<index_t> cursor(rows.row_ptr.begin(), rows.row_ptr.end() - 1);
        for (size_t n = 0; n < pattern.row_indices.size(); ++n) {
            index_t i = pattern.row_indices[n];
            index_t j = pattern.col_indices[n];
            if (in_range(i, j)) {
                rows.cols[cursor[i]++] = j;
            }
        }

        // Sort each row and drop duplicate keys, which would otherwise be
        // counted twice in the softmax denominator.
        index_t write = 0;
        index_t row_begin = 0;
        for (index_t i = 0; i < seq_len; ++i) {
            const index_t row_end = rows.row_ptr[i + 1];
            auto first = rows.cols.begin() + row_begin;
            auto last = rows.cols.begin() + row_end;
            std::sort(first, last);
            last = std::unique(first, last);
            const index_t unique_len = static_cast<index_t>(last - first);
            std::move(first, last, rows.cols.begin() + write);
            rows.row_ptr[i] = write;
            write += unique_len;
            row_begin = row_end;
        }
        rows.row_ptr[seq_len] = write;
        rows.cols.resize(static_cast<size_t>(write));
        return rows;
    }

    SparseTensor SparseAttention::get_attention_pattern(const TensorShape& shape) const {
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
    SparseTensor SlidingWindowAttention::generate_sparse_pattern(const TensorShape& shape) const {
        SparseTensor pattern(shape, device_);
        index_t seq_len = shape.sequence_length;
        index_t window_size = config_.window_size;
        
        // Estimate capacity: each position attends to at most 2*window_size+1 positions
        index_t estimated_nnz = seq_len * std::min(2 * window_size + 1, seq_len);
        pattern.reserve(estimated_nnz);
        
        for (index_t i = 0; i < seq_len; ++i) {
            index_t start = std::max(static_cast<index_t>(0), i - window_size);
            index_t end = std::min(seq_len, i + window_size + 1);
            
            for (index_t j = start; j < end; ++j) {
                pattern.add_entry(i, j, 1.0f);
            }
        }
        
        return pattern;
    }

    // Block Sparse Attention
    SparseTensor BlockSparseAttention::generate_sparse_pattern(const TensorShape& shape) const {
        SparseTensor pattern(shape, device_);
        index_t seq_len = shape.sequence_length;
        index_t block_size = config_.block_size;
        
        // Number of blocks
        index_t num_blocks = (seq_len + block_size - 1) / block_size;
        
        // Each block attends to itself and adjacent blocks
        for (index_t block_i = 0; block_i < num_blocks; ++block_i) {
            for (index_t block_j = 0; block_j < num_blocks; ++block_j) {
                // Allow attention within the same block and adjacent blocks
                if (std::abs(static_cast<int64_t>(block_i) - static_cast<int64_t>(block_j)) <= 1) {
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
    SparseTensor LongformerAttention::generate_sparse_pattern(const TensorShape& shape) const {
        SparseTensor pattern(shape, device_);
        index_t seq_len = shape.sequence_length;
        index_t window_size = config_.window_size;
        index_t num_global = config_.num_global_tokens;
        
        const index_t global_end = std::min(num_global, seq_len);

        for (index_t i = 0; i < seq_len; ++i) {
            // Global tokens attend to every position
            if (i < global_end) {
                for (index_t j = 0; j < seq_len; ++j) {
                    pattern.add_entry(i, j, 1.0f);
                }
                continue;
            }

            // Every other token attends to the global tokens and its local
            // window. Global tokens inside the window are added only once.
            index_t start = std::max(static_cast<index_t>(0), i - window_size);
            index_t end = std::min(seq_len, i + window_size + 1);
            for (index_t j = 0; j < std::min(global_end, start); ++j) {
                pattern.add_entry(i, j, 1.0f);
            }
            for (index_t j = start; j < end; ++j) {
                pattern.add_entry(i, j, 1.0f);
            }
        }

        return pattern;
    }

    // Financial Attention
    SparseTensor FinancialAttention::generate_sparse_pattern(const TensorShape& shape) const {
        SparseTensor pattern(shape, device_);
        index_t seq_len = shape.sequence_length;

        index_t local_window = config_.local_window_size;
        index_t stride = config_.dilation_stride;
        index_t cluster_size = config_.dilation_cluster_size;
        index_t num_clusters = config_.dilation_num_clusters;

        for (index_t i = 0; i < seq_len; ++i) {
            // Local causal window: [start_local, i]
            index_t start_local = (i >= local_window) ? (i - local_window + 1) : static_cast<index_t>(0);
            for (index_t j = start_local; j <= i; ++j) {
                pattern.add_entry(i, j, 1.0f);
            }

            // Dilated clusters — skip any position already covered by the local window
            // to prevent duplicate entries that would corrupt softmax normalization.
            for (index_t k = 1; k <= num_clusters; ++k) {
                int64_t cluster_end = static_cast<int64_t>(i) - static_cast<int64_t>(k) * static_cast<int64_t>(stride);
                int64_t cluster_start = cluster_end - static_cast<int64_t>(cluster_size) + 1;
                if (cluster_end < 0) break;
                if (cluster_start < 0) cluster_start = 0;
                for (int64_t j = cluster_start; j <= cluster_end; ++j) {
                    if (static_cast<index_t>(j) >= start_local) continue; // already in local window
                    pattern.add_entry(i, static_cast<index_t>(j), 1.0f);
                }
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
            return attention.get_attention_pattern(shape);
        }

        SparseTensor create_block_sparse_pattern(const TensorShape& shape, index_t block_size) {
            BlockSparseAttention attention(block_size);
            return attention.get_attention_pattern(shape);
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
