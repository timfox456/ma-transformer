// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <chrono>
#include <climits>
#include <set>
#include "../../src/csrc/ma_core.hpp"
#include "../../src/csrc/attention_interface.hpp"

// Test fixture for ma_core tests
class MaCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code for each test
    }

    void TearDown() override {
        // Cleanup code for each test
    }
};

// Basic arithmetic tests
TEST_F(MaCoreTest, TestAddPositiveNumbers) {
    EXPECT_EQ(ma_core::add(2, 3), 5);
    EXPECT_EQ(ma_core::add(10, 15), 25);
    EXPECT_EQ(ma_core::add(100, 200), 300);
}

TEST_F(MaCoreTest, TestAddNegativeNumbers) {
    EXPECT_EQ(ma_core::add(-2, -3), -5);
    EXPECT_EQ(ma_core::add(-10, 5), -5);
    EXPECT_EQ(ma_core::add(10, -15), -5);
}

TEST_F(MaCoreTest, TestAddZero) {
    EXPECT_EQ(ma_core::add(0, 0), 0);
    EXPECT_EQ(ma_core::add(5, 0), 5);
    EXPECT_EQ(ma_core::add(0, -5), -5);
}

TEST_F(MaCoreTest, TestAddBoundaryValues) {
    // Test with integer limits
    EXPECT_EQ(ma_core::add(1, -1), 0);
    EXPECT_EQ(ma_core::add(INT_MAX, 0), INT_MAX);
    EXPECT_EQ(ma_core::add(INT_MIN, 0), INT_MIN);
}

// Commutative property test
TEST_F(MaCoreTest, TestAddCommutative) {
    int a = 42, b = 17;
    EXPECT_EQ(ma_core::add(a, b), ma_core::add(b, a));
}

// Associative property test (for future multi-operand operations)
TEST_F(MaCoreTest, TestAddAssociative) {
    int a = 10, b = 20, c = 30;
    EXPECT_EQ(ma_core::add(ma_core::add(a, b), c), ma_core::add(a, ma_core::add(b, c)));
}

// Performance test for basic operations
TEST_F(MaCoreTest, TestAddPerformance) {
    const int iterations = 1000000;
    auto start = std::chrono::high_resolution_clock::now();
    
    volatile int result = 0;  // volatile to prevent optimization
    for (int i = 0; i < iterations; ++i) {
        result = ma_core::add(i, i + 1);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete within reasonable time (less than 100ms for 1M operations)
    EXPECT_LT(duration.count(), 100000);
    EXPECT_EQ(result, ma_core::add(iterations - 1, iterations));  // Check last result
}

// Attention tests
TEST_F(MaCoreTest, TestFinancialAttentionPattern) {
    ma_core::AttentionConfig config(ma_core::AttentionPattern::FINANCIAL);
    config.local_window_size = 5;
    config.dilation_stride = 10;
    config.dilation_cluster_size = 2;
    config.dilation_num_clusters = 1;
    
    ma_core::TensorShape shape(1, 20, 1, 1);
    auto attention = ma_core::create_attention(config, ma_core::Device::CPU);
    auto pattern = attention->generate_sparse_pattern(shape);
    
    // Query 15:
    // Local window: [15-5+1, 15] = [11, 15]
    // Dilated cluster 1: [15-10-2+1, 15-10] = [4, 5]
    
    std::set<ma_core::index_t> expected = {11, 12, 13, 14, 15, 4, 5};
    std::set<ma_core::index_t> actual;
    
    for (size_t i = 0; i < pattern.row_indices.size(); ++i) {
        if (pattern.row_indices[i] == 15) {
            actual.insert(pattern.col_indices[i]);
        }
    }
    
    EXPECT_EQ(actual, expected);
}
