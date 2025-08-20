#pragma once

#include "attention_types.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace ma_core {

    /**
     * Lightweight tensor class for cross-platform operations
     * Supports both dense and sparse tensor operations
     */
    class Tensor {
    private:
        std::unique_ptr<scalar_t[]> data_;
        TensorShape shape_;
        Device device_;
        MemoryLayout layout_;
        bool owns_data_;

    public:
        // Constructors
        Tensor(const TensorShape& shape, Device device = Device::CPU, 
               MemoryLayout layout = MemoryLayout::NWHD);
        
        Tensor(scalar_t* data, const TensorShape& shape, Device device = Device::CPU,
               MemoryLayout layout = MemoryLayout::NWHD, bool owns_data = false);
        
        // Copy constructor and assignment
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        
        // Move constructor and assignment
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        
        ~Tensor() = default;

        // Accessors
        scalar_t* data() { return data_.get(); }
        const scalar_t* data() const { return data_.get(); }
        const TensorShape& shape() const { return shape_; }
        Device device() const { return device_; }
        MemoryLayout layout() const { return layout_; }
        
        index_t size() const { return shape_.total_elements(); }
        size_t nbytes() const { return static_cast<size_t>(size()) * sizeof(scalar_t); }
        bool empty() const { return size() == 0; }

        // Element access (for debugging/small tensors)
        scalar_t& operator[](index_t idx);
        const scalar_t& operator[](index_t idx) const;
        
        // Multi-dimensional access [batch][seq][head][dim]
        scalar_t& at(index_t batch, index_t seq, index_t head, index_t dim);
        const scalar_t& at(index_t batch, index_t seq, index_t head, index_t dim) const;

        // Tensor operations
        Tensor copy() const;
        Tensor to_device(Device target_device) const;
        void fill(scalar_t value);
        void zero();

        // Bulk memory copy operations (host memory for now)
        void copy_from(const void* src, size_t bytes) {
            if (bytes != nbytes()) {
                throw std::runtime_error("copy_from: size mismatch");
            }
            std::memcpy(data_.get(), src, bytes);
        }
        void copy_to(void* dst, size_t bytes) const {
            if (bytes != nbytes()) {
                throw std::runtime_error("copy_to: size mismatch");
            }
            std::memcpy(dst, data_.get(), bytes);
        }

        // Shape operations
        Tensor reshape(const TensorShape& new_shape) const;
        Tensor transpose(int dim0, int dim1) const;
        
        // View operations (no data copy)
        Tensor view(const TensorShape& new_shape) const;
        Tensor slice(index_t start_seq, index_t end_seq) const;

    private:
        index_t compute_index(index_t batch, index_t seq, index_t head, index_t dim) const;
        void allocate_memory();
        void copy_data(const Tensor& other);
    };

    /**
     * Sparse tensor representation for sparse attention patterns
     */
    struct SparseTensor {
        std::vector<index_t> row_indices;
        std::vector<index_t> col_indices;
        std::vector<scalar_t> values;
        TensorShape shape;
        Device device;

        SparseTensor(const TensorShape& shape, Device device = Device::CPU)
            : shape(shape), device(device) {}

        index_t nnz() const { return values.size(); } // Number of non-zeros
        bool empty() const { return values.empty(); }

        void reserve(index_t capacity) {
            row_indices.reserve(capacity);
            col_indices.reserve(capacity);
            values.reserve(capacity);
        }

        void add_entry(index_t row, index_t col, scalar_t value) {
            row_indices.push_back(row);
            col_indices.push_back(col);
            values.push_back(value);
        }

        void clear() {
            row_indices.clear();
            col_indices.clear();
            values.clear();
        }
    };

    // Factory functions for common tensor patterns
    Tensor zeros(const TensorShape& shape, Device device = Device::CPU);
    Tensor ones(const TensorShape& shape, Device device = Device::CPU);
    Tensor random_normal(const TensorShape& shape, scalar_t mean = 0.0f, 
                        scalar_t std = 1.0f, Device device = Device::CPU);
    
    // Tensor operations
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor multiply(const Tensor& a, scalar_t scalar);
    Tensor softmax(const Tensor& input, int dim = -1);

} // namespace ma_core
