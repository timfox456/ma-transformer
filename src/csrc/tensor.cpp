// SPDX-License-Identifier: Apache-2.0
#include "tensor.hpp"
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>

namespace ma_core {

    // Tensor implementation
    Tensor::Tensor(const TensorShape& shape, Device device, MemoryLayout layout)
        : shape_(shape), device_(device), layout_(layout), owns_data_(true) {
        allocate_memory();
    }

    Tensor::Tensor(scalar_t* data, const TensorShape& shape, Device device, 
                   MemoryLayout layout, bool owns_data)
        : data_(owns_data ? nullptr : data), shape_(shape), device_(device), 
          layout_(layout), owns_data_(owns_data) {
        if (owns_data) {
            allocate_memory();
            std::memcpy(data_.get(), data, size() * sizeof(scalar_t));
        } else {
            data_.reset(data);
        }
    }

    Tensor::Tensor(const Tensor& other)
        : shape_(other.shape_), device_(other.device_), 
          layout_(other.layout_), owns_data_(true) {
        allocate_memory();
        copy_data(other);
    }

    Tensor& Tensor::operator=(const Tensor& other) {
        if (this != &other) {
            shape_ = other.shape_;
            device_ = other.device_;
            layout_ = other.layout_;
            owns_data_ = true;
            allocate_memory();
            copy_data(other);
        }
        return *this;
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : data_(std::move(other.data_)), shape_(other.shape_), 
          device_(other.device_), layout_(other.layout_), 
          owns_data_(other.owns_data_) {
        other.owns_data_ = false;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            shape_ = other.shape_;
            device_ = other.device_;
            layout_ = other.layout_;
            owns_data_ = other.owns_data_;
            other.owns_data_ = false;
        }
        return *this;
    }

    scalar_t& Tensor::operator[](index_t idx) {
        if (idx >= size()) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data_[idx];
    }

    const scalar_t& Tensor::operator[](index_t idx) const {
        if (idx >= size()) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data_[idx];
    }

    scalar_t& Tensor::at(index_t batch, index_t seq, index_t head, index_t dim) {
        index_t idx = compute_index(batch, seq, head, dim);
        return data_[idx];
    }

    const scalar_t& Tensor::at(index_t batch, index_t seq, index_t head, index_t dim) const {
        index_t idx = compute_index(batch, seq, head, dim);
        return data_[idx];
    }

    void Tensor::fill(scalar_t value) {
        for (index_t i = 0; i < size(); ++i) {
            data_[i] = value;
        }
    }

    void Tensor::zero() {
        fill(0.0f);
    }

    Tensor Tensor::copy() const {
        return Tensor(*this);
    }

    index_t Tensor::compute_index(index_t batch, index_t seq, index_t head, index_t dim) const {
        // Default layout: NWHD [batch, seq, heads, dim]
        switch (layout_) {
            case MemoryLayout::NWHD:
                return ((batch * shape_.sequence_length + seq) * shape_.num_heads + head) * shape_.head_dim + dim;
            case MemoryLayout::NHWD:
                return ((batch * shape_.num_heads + head) * shape_.sequence_length + seq) * shape_.head_dim + dim;
            default:
                throw std::runtime_error("Unsupported memory layout");
        }
    }

    void Tensor::allocate_memory() {
        if (size() > 0) {
            data_ = std::make_unique<scalar_t[]>(size());
        }
    }

    void Tensor::copy_data(const Tensor& other) {
        if (size() != other.size()) {
            throw std::runtime_error("Cannot copy tensors of different sizes");
        }
        std::memcpy(data_.get(), other.data_.get(), size() * sizeof(scalar_t));
    }

    // Factory functions
    Tensor zeros(const TensorShape& shape, Device device) {
        Tensor tensor(shape, device);
        tensor.zero();
        return tensor;
    }

    Tensor ones(const TensorShape& shape, Device device) {
        Tensor tensor(shape, device);
        tensor.fill(1.0f);
        return tensor;
    }

    Tensor random_normal(const TensorShape& shape, scalar_t mean, scalar_t std, Device device) {
        Tensor tensor(shape, device);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<scalar_t> dist(mean, std);
        
        for (index_t i = 0; i < tensor.size(); ++i) {
            tensor[i] = dist(gen);
        }
        
        return tensor;
    }

    // Basic tensor operations (CPU implementations first)
    Tensor add(const Tensor& a, const Tensor& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Cannot add tensors of different sizes");
        }
        
        Tensor result(a.shape(), a.device(), a.layout());
        for (index_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    Tensor multiply(const Tensor& a, scalar_t scalar) {
        Tensor result(a.shape(), a.device(), a.layout());
        for (index_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] * scalar;
        }
        return result;
    }

    Tensor softmax(const Tensor& input, int dim) {
        // Simple 2D softmax implementation for attention scores
        // Assumes input is [seq_len, seq_len] attention matrix
        if (input.shape().batch_size != 1 || input.shape().num_heads != 1) {
            throw std::runtime_error("Softmax currently only supports 2D tensors");
        }
        
        Tensor result(input.shape(), input.device(), input.layout());
        index_t seq_len = input.shape().sequence_length;
        index_t head_dim = input.shape().head_dim;
        
        for (index_t i = 0; i < seq_len; ++i) {
            // Find max for numerical stability
            scalar_t max_val = input.at(0, i, 0, 0);
            for (index_t j = 1; j < head_dim; ++j) {
                max_val = std::max(max_val, input.at(0, i, 0, j));
            }
            
            // Compute exp and sum
            scalar_t sum = 0.0f;
            for (index_t j = 0; j < head_dim; ++j) {
                scalar_t exp_val = std::exp(input.at(0, i, 0, j) - max_val);
                result.at(0, i, 0, j) = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (index_t j = 0; j < head_dim; ++j) {
                result.at(0, i, 0, j) /= sum;
            }
        }
        
        return result;
    }

    // Basic matrix multiplication (naive implementation for now)
    Tensor matmul(const Tensor& a, const Tensor& b) {
        // Simplified 2D matrix multiplication
        // a: [seq_len, head_dim], b: [head_dim, seq_len]
        if (a.shape().head_dim != b.shape().sequence_length) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }
        
        TensorShape result_shape(a.shape().batch_size, a.shape().sequence_length, 
                               a.shape().num_heads, b.shape().head_dim);
        Tensor result(result_shape, a.device(), a.layout());
        result.zero();
        
        index_t M = a.shape().sequence_length;
        index_t N = b.shape().head_dim;
        index_t K = a.shape().head_dim;
        
        for (index_t i = 0; i < M; ++i) {
            for (index_t j = 0; j < N; ++j) {
                for (index_t k = 0; k < K; ++k) {
                    result.at(0, i, 0, j) += a.at(0, i, 0, k) * b.at(0, k, 0, j);
                }
            }
        }
        
        return result;
    }

} // namespace ma_core
