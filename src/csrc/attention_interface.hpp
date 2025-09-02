// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tensor.hpp"
#include "attention_types.hpp"
#include <memory>

namespace ma_core {

    /**
     * Abstract base class for all attention mechanisms
     * Provides unified interface for both sparse and dense attention
     */
    class AttentionBase {
    public:
        AttentionBase(const AttentionConfig& config, Device device = Device::CPU)
            : config_(config), device_(device) {}
        
        virtual ~AttentionBase() = default;

        // Pure virtual functions that must be implemented
        virtual Tensor forward(const Tensor& query, const Tensor& key, 
                              const Tensor& value) = 0;
        
        virtual SparseTensor get_attention_pattern(const TensorShape& shape) = 0;
        
        // Optional: backward pass for training (can be implemented later)
        virtual std::tuple<Tensor, Tensor, Tensor> backward(
            const Tensor& grad_output,
            const Tensor& query, 
            const Tensor& key, 
            const Tensor& value) {
            throw std::runtime_error("Backward pass not implemented for this attention type");
        }

        // Configuration accessors
        const AttentionConfig& config() const { return config_; }
        Device device() const { return device_; }
        
        // Utility functions
        virtual index_t memory_usage(const TensorShape& shape) const = 0;
        virtual bool supports_device(Device device) const = 0;

    protected:
        AttentionConfig config_;
        Device device_;

        // Helper functions for all attention types
        Tensor apply_causal_mask(const Tensor& attention_scores) const;
        Tensor apply_attention_dropout(const Tensor& attention_weights) const;
        Tensor scale_query(const Tensor& query) const;
    };

    /**
     * Dense attention base class
     * Handles full and causal attention patterns
     */
    class DenseAttention : public AttentionBase {
    public:
        DenseAttention(const AttentionConfig& config, Device device = Device::CPU)
            : AttentionBase(config, device) {}
        
        Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value) override;
        SparseTensor get_attention_pattern(const TensorShape& shape) override;
        index_t memory_usage(const TensorShape& shape) const override;
        bool supports_device(Device device) const override;

    protected:
        virtual Tensor compute_attention_scores(const Tensor& query, const Tensor& key);
        virtual Tensor apply_attention_mask(const Tensor& attention_scores);
    };

    /**
     * Sparse attention base class
     * Handles various sparse attention patterns
     */
    class SparseAttention : public AttentionBase {
    public:
        SparseAttention(const AttentionConfig& config, Device device = Device::CPU)
            : AttentionBase(config, device) {}
        
        Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value) override;
        SparseTensor get_attention_pattern(const TensorShape& shape) override;
        index_t memory_usage(const TensorShape& shape) const override;
        bool supports_device(Device device) const override;

    protected:
        virtual SparseTensor generate_sparse_pattern(const TensorShape& shape) = 0;
        virtual Tensor compute_sparse_attention_scores(const Tensor& query, const Tensor& key,
                                                      const SparseTensor& pattern);
        Tensor apply_sparse_softmax(const Tensor& scores, const SparseTensor& pattern);
        Tensor compute_sparse_attention_output(const Tensor& attention_weights, 
                                             const Tensor& value, 
                                             const SparseTensor& pattern);
    };

    /**
     * Specific sparse attention implementations
     */
    class SlidingWindowAttention : public SparseAttention {
    public:
        SlidingWindowAttention(index_t window_size, Device device = Device::CPU)
            : SparseAttention(AttentionConfig(AttentionPattern::SLIDING_WINDOW), device) {
            config_.window_size = window_size;
        }

    protected:
        SparseTensor generate_sparse_pattern(const TensorShape& shape) override;
    };

    class BlockSparseAttention : public SparseAttention {
    public:
        BlockSparseAttention(index_t block_size, Device device = Device::CPU)
            : SparseAttention(AttentionConfig(AttentionPattern::BLOCK_SPARSE), device) {
            config_.block_size = block_size;
        }

    protected:
        SparseTensor generate_sparse_pattern(const TensorShape& shape) override;
    };

    class LongformerAttention : public SparseAttention {
    public:
        LongformerAttention(index_t window_size, index_t num_global_tokens, Device device = Device::CPU)
            : SparseAttention(AttentionConfig(AttentionPattern::LONGFORMER), device) {
            config_.window_size = window_size;
            config_.num_global_tokens = num_global_tokens;
        }

    protected:
        SparseTensor generate_sparse_pattern(const TensorShape& shape) override;
    };

    /**
     * Factory function for creating attention mechanisms
     */
    std::unique_ptr<AttentionBase> create_attention(const AttentionConfig& config, 
                                                   Device device = Device::CPU);

    /**
     * Utility functions for attention operations
     */
    namespace attention_utils {
        // Pattern generation utilities
        SparseTensor create_causal_pattern(const TensorShape& shape);
        SparseTensor create_sliding_window_pattern(const TensorShape& shape, index_t window_size);
        SparseTensor create_block_sparse_pattern(const TensorShape& shape, index_t block_size);
        SparseTensor create_random_sparse_pattern(const TensorShape& shape, scalar_t sparsity_ratio);
        
        // Performance utilities
        index_t estimate_flops(const AttentionConfig& config, const TensorShape& shape);
        index_t estimate_memory_usage(const AttentionConfig& config, const TensorShape& shape);
        
        // Debugging utilities
        void print_attention_pattern(const SparseTensor& pattern);
        bool validate_attention_pattern(const SparseTensor& pattern, const TensorShape& shape);
    }

} // namespace ma_core
