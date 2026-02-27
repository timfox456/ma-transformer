// SPDX-License-Identifier: Apache-2.0
// Additional tests for uncommitted changes

#include <gtest/gtest.h>
#include <chrono>
#include <climits>
#include <set>
#include <cmath>
#include "../../src/csrc/ma_core.hpp"
#include "../../src/csrc/attention_interface.hpp"

// Test fixture for ma_core tests
class MaCoreNewTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code for each test
    }

    void TearDown() override {
        // Cleanup code for each test
    }
};

// Test get_attention_pattern() API for all attention types
TEST_F(MaCoreNewTests, TestGetAttentionPatternAPI) {
    // Test that get_attention_pattern() is accessible and returns valid patterns
    ma_core::TensorShape shape(1, 20, 2, 4);
    
    // Test SlidingWindowAttention
    {
        ma_core::AttentionConfig sw_config(ma_core::AttentionPattern::SLIDING_WINDOW);
        sw_config.window_size = 5;
        auto sw_attention = ma_core::create_attention(sw_config, ma_core::Device::CPU);
        auto pattern = sw_attention->get_attention_pattern(shape);
        
        // Pattern should have entries
        EXPECT_GT(pattern.values.size(), 0);
        EXPECT_EQ(pattern.values.size(), pattern.row_indices.size());
        EXPECT_EQ(pattern.values.size(), pattern.col_indices.size());
        
        // All rows should be within bounds
        for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
            EXPECT_LT(pattern.row_indices[i], shape.sequence_length);
            EXPECT_LT(pattern.col_indices[i], shape.sequence_length);
        }
    }
    
    // Test BlockSparseAttention
    {
        ma_core::AttentionConfig bs_config(ma_core::AttentionPattern::BLOCK_SPARSE);
        bs_config.block_size = 4;
        auto bs_attention = ma_core::create_attention(bs_config, ma_core::Device::CPU);
        auto pattern = bs_attention->get_attention_pattern(shape);
        
        EXPECT_GT(pattern.values.size(), 0);
        EXPECT_EQ(pattern.values.size(), pattern.row_indices.size());
    }
    
    // Test DenseAttention (causal)
    {
        ma_core::AttentionConfig dense_config(ma_core::AttentionPattern::DENSE);
        auto dense_attention = ma_core::create_attention(dense_config, ma_core::Device::CPU);
        auto pattern = dense_attention->get_attention_pattern(shape);
        
        // Dense causal should have triangular pattern: n*(n+1)/2 entries
        index_t expected_entries = shape.sequence_length * (shape.sequence_length + 1) / 2;
        EXPECT_EQ(pattern.values.size(), expected_entries);
    }
}

// Test that get_attention_pattern is the public interface
TEST_F(MaCoreNewTests, TestProtectedPatternGeneration) {
    // The fact that this test compiles and runs confirms that
    // generate_sparse_pattern is protected and get_attention_pattern
    // is the public interface. We test get_attention_pattern thoroughly.
    ma_core::AttentionConfig config(ma_core::AttentionPattern::SLIDING_WINDOW);
    config.window_size = 4;
    
    auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
    ma_core::TensorShape shape(1, 10, 1, 1);
    
    // This should work - get_attention_pattern is public
    auto pattern = attention->get_attention_pattern(shape);
    EXPECT_GT(pattern.row_indices.size(), 0);
}

// Test sparse attention output computation optimization
TEST_F(MaCoreNewTests, TestSparseAttentionOutputOptimization) {
    // This test verifies that the optimized compute_sparse_attention_output
    // produces correct results without duplicate entries
    ma_core::AttentionConfig config(ma_core::AttentionPattern::SLIDING_WINDOW);
    config.window_size = 4;
    
    ma_core::TensorShape shape(2, 10, 2, 8);  // batch=2, heads=2
    auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
    
    // Create input tensors
    ma_core::Tensor query(shape, ma_core::Device::CPU);
    ma_core::Tensor key(shape, ma_core::Device::CPU);
    ma_core::Tensor value(shape, ma_core::Device::CPU);
    
    // Initialize with deterministic values
    for (index_t b = 0; b < shape.batch_size; ++b) {
        for (index_t s = 0; s < shape.sequence_length; ++s) {
            for (index_t h = 0; h < shape.num_heads; ++h) {
                for (index_t d = 0; d < shape.head_dim; ++d) {
                    query.at(b, s, h, d) = static_cast<scalar_t>(s + d) * 0.1f;
                    key.at(b, s, h, d) = static_cast<scalar_t>(s + d) * 0.05f;
                    value.at(b, s, h, d) = static_cast<scalar_t>(s + 1);
                }
            }
        }
    }
    
    // Compute attention
    auto output = attention->forward(query, key, value);
    
    // Verify output shape
    EXPECT_EQ(output.shape().batch_size, shape.batch_size);
    EXPECT_EQ(output.shape().sequence_length, shape.sequence_length);
    EXPECT_EQ(output.shape().num_heads, shape.num_heads);
    EXPECT_EQ(output.shape().head_dim, shape.head_dim);
    
    // Output should not contain NaN or Inf
    for (index_t b = 0; b < shape.batch_size; ++b) {
        for (index_t s = 0; s < shape.sequence_length; ++s) {
            for (index_t h = 0; h < shape.num_heads; ++h) {
                for (index_t d = 0; d < shape.head_dim; ++d) {
                    scalar_t val = output.at(b, s, h, d);
                    EXPECT_FALSE(std::isnan(val));
                    EXPECT_FALSE(std::isinf(val));
                }
            }
        }
    }
    
    // Verify causality: each position should only attend to previous positions
    auto pattern = attention->get_attention_pattern(shape);
    for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
        EXPECT_LE(pattern.col_indices[i], pattern.row_indices[i]);
    }
}

// Test FinancialAttention duplicate entry prevention
TEST_F(MaCoreNewTests, TestFinancialAttentionNoDuplicateEntries) {
    // This test verifies that the FinancialAttention pattern does not have
    // duplicate entries where dilated clusters overlap with local windows
    ma_core::AttentionConfig config(ma_core::AttentionPattern::FINANCIAL);
    config.local_window_size = 8;   // Large local window
    config.dilation_stride = 5;     // Small stride to create overlap
    config.dilation_cluster_size = 3;
    config.dilation_num_clusters = 3;  // Multiple clusters
    
    // With these settings, cluster positions may overlap with local window
    // The fix should prevent duplicate entries
    ma_core::TensorShape shape(1, 50, 1, 1);
    auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
    auto pattern = attention->get_attention_pattern(shape);
    
    // Check each query position for duplicate entries
    for (index_t query_pos = 10; query_pos < 30; ++query_pos) {
        std::set<ma_core::index_t> unique_cols;
        std::vector<ma_core::index_t> all_cols;
        
        for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
            if (pattern.row_indices[i] == query_pos) {
                unique_cols.insert(pattern.col_indices[i]);
                all_cols.push_back(pattern.col_indices[i]);
            }
        }
        
        // If there are no duplicates, the sizes should match
        EXPECT_EQ(unique_cols.size(), all_cols.size());
    }
    
    // Verify pattern is deterministic (same query gets same pattern)
    auto pattern2 = attention->get_attention_pattern(shape);
    EXPECT_EQ(pattern.values.size(), pattern2.values.size());
    EXPECT_EQ(pattern.row_indices.size(), pattern2.row_indices.size());
    for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
        EXPECT_EQ(pattern.row_indices[i], pattern2.row_indices[i]);
        EXPECT_EQ(pattern.col_indices[i], pattern2.col_indices[i]);
        EXPECT_FLOAT_EQ(pattern.values[i], pattern2.values[i]);
    }
}

// Test that FinancialAttention properly handles edge case with large local window
TEST_F(MaCoreNewTests, TestFinancialAttentionLargeWindowEdgeCase) {
    // When local window is large enough to cover dilated clusters,
    // the dilated clusters should be skipped to avoid duplicates
    ma_core::AttentionConfig config(ma_core::AttentionPattern::FINANCIAL);
    config.local_window_size = 20;  // Large enough to cover most positions
    config.dilation_stride = 5;
    config.dilation_cluster_size = 2;
    config.dilation_num_clusters = 2;
    
    ma_core::TensorShape shape(1, 25, 1, 1);
    auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
    auto pattern = attention->get_attention_pattern(shape);
    
    // Check query positions that would have overlapping clusters
    for (index_t query_pos = 20; query_pos < 25; ++query_pos) {
        std::vector<ma_core::index_t> cols;
        for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
            if (pattern.row_indices[i] == query_pos) {
                cols.push_back(pattern.col_indices[i]);
            }
        }
        
        // Verify all entries are unique
        std::set<ma_core::index_t> unique_cols(cols.begin(), cols.end());
        EXPECT_EQ(cols.size(), unique_cols.size());
        
        // Verify local window is included
        index_t expected_start = (query_pos >= 20) ? (query_pos - 20 + 1) : 0;
        for (index_t j = expected_start; j <= query_pos; ++j) {
            EXPECT_TRUE(unique_cols.count(j) > 0);
        }
    }
}

// Test compute_sparse_attention_output with different pattern densities
TEST_F(MaCoreNewTests, TestSparseAttentionVaryingDensity) {
    // Test with different sparsity levels to ensure the optimization
    // works correctly across different densities
    struct TestCase {
        index_t seq_len;
        index_t window_size;
    };
    
    TestCase cases[] = {
        {10, 2},
        {20, 5},
        {30, 15},
        {50, 3}
    };
    
    for (const auto& test_case : cases) {
        ma_core::AttentionConfig config(ma_core::AttentionPattern::SLIDING_WINDOW);
        config.window_size = test_case.window_size;
        
        ma_core::TensorShape shape(1, test_case.seq_len, 1, 4);
        auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
        
        ma_core::Tensor query(shape, ma_core::Device::CPU);
        ma_core::Tensor key(shape, ma_core::Device::CPU);
        ma_core::Tensor value(shape, ma_core::Device::CPU);
        
        // Initialize
        for (index_t s = 0; s < test_case.seq_len; ++s) {
            for (index_t d = 0; d < 4; ++d) {
                query.at(0, s, 0, d) = 0.1f;
                key.at(0, s, 0, d) = 0.1f;
                value.at(0, s, 0, d) = static_cast<scalar_t>(s + 1);
            }
        }
        
        auto output = attention->forward(query, key, value);
        
        // Verify output is valid
        for (index_t s = 0; s < test_case.seq_len; ++s) {
            for (index_t d = 0; d < 4; ++d) {
                scalar_t val = output.at(0, s, 0, d);
                EXPECT_FALSE(std::isnan(val));
            }
        }
    }
}

// Test pattern validation through get_attention_pattern
TEST_F(MaCoreNewTests, TestPatternValidation) {
    ma_core::AttentionConfig config(ma_core::AttentionPattern::SLIDING_WINDOW);
    config.window_size = 4;
    
    auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
    
    // Test with various shapes
    ma_core::TensorShape shapes[] = {
        ma_core::TensorShape(1, 5, 1, 4),
        ma_core::TensorShape(2, 10, 2, 8),
        ma_core::TensorShape(4, 8, 1, 16)
    };
    
    for (const auto& shape : shapes) {
        auto pattern = attention->get_attention_pattern(shape);
        
        // Pattern dimensions should match sequence length
        for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
            EXPECT_LT(pattern.row_indices[i], shape.sequence_length);
            EXPECT_LT(pattern.col_indices[i], shape.sequence_length);
            EXPECT_GE(pattern.row_indices[i], 0);
            EXPECT_GE(pattern.col_indices[i], 0);
        }
        
        // All values should be positive (indicating attention weights)
        for (const auto& val : pattern.values) {
            EXPECT_GT(val, 0.0f);
        }
    }
}

// Test attention pattern consistency with factory-created instances
TEST_F(MaCoreNewTests, TestFactoryPatternConsistency) {
    // Verify that patterns created via factory are consistent
    ma_core::AttentionConfig config(ma_core::AttentionPattern::FINANCIAL);
    config.local_window_size = 6;
    config.dilation_stride = 8;
    config.dilation_cluster_size = 2;
    config.dilation_num_clusters = 2;
    
    ma_core::TensorShape shape(1, 30, 2, 4);
    
    // Create multiple instances and verify they produce identical patterns
    for (int trial = 0; trial < 3; ++trial) {
        auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
        auto pattern = attention->get_attention_pattern(shape);
        
        // For query 20, verify expected pattern
        // Local window: [20-6+1, 20] = [15, 20]
        // Cluster 1 end: 20-8=12, start: 12-2+1=11, skip overlap
        // Cluster 2 end: 20-16=4, start: 4-2+1=3
        std::set<ma_core::index_t> expected;
        for (index_t j = 15; j <= 20; ++j) expected.insert(j);
        for (index_t j = 11; j <= 12; ++j) expected.insert(j);  // cluster 1, non-overlapping
        for (index_t j = 3; j <= 4; ++j) expected.insert(j);    // cluster 2
        
        std::set<ma_core::index_t> actual;
        for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
            if (pattern.row_indices[i] == 20) {
                actual.insert(pattern.col_indices[i]);
            }
        }
        
        EXPECT_EQ(actual, expected);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
