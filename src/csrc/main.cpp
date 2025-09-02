// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <array>
#include "ma_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ma_core, m) {
    m.doc() = "MA Transformer Core C++ Engine";
    
    // Legacy function
    m.def("add", [](int64_t a, int64_t b) { return ma_core::add(a, b); }, "A function which adds two numbers");
    
    // Basic tensor functionality
    m.def("create_tensor", &ma_core::create_tensor, 
          "Create a tensor with given shape",
          py::arg("shape"), py::arg("device") = "cpu");
    
    m.def("create_random_tensor", &ma_core::create_random_tensor,
          "Create a tensor filled with random values from normal distribution",
          py::arg("shape"), py::arg("mean") = 0.0f, py::arg("std") = 1.0f, py::arg("device") = "cpu");
    
    // Sparse pattern functionality
    m.def("create_sliding_window_pattern", &ma_core::create_sliding_window_pattern,
          "Create a sliding window sparse attention pattern",
          py::arg("seq_len"), py::arg("window_size"));
    
    m.def("get_pattern_statistics", &ma_core::get_pattern_statistics,
          "Get statistics about a sparse attention pattern");
    
    // Attention computation functions
    m.def("compute_dense_attention", &ma_core::compute_dense_attention,
          "Compute dense attention (Q@K^T, softmax, @V)",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("use_causal_mask") = false);
    
    m.def("compute_sparse_attention", &ma_core::compute_sparse_attention,
          "Compute sparse attention with sliding window",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("window_size") = 64);
    
    // SparseTensor class
    py::class_<ma_core::SparseTensor>(m, "SparseTensor")
        .def_readonly("shape", &ma_core::SparseTensor::shape)
        .def_readonly("device", &ma_core::SparseTensor::device)
        .def("nnz", &ma_core::SparseTensor::nnz, "Number of non-zero elements")
        .def("empty", &ma_core::SparseTensor::empty, "Check if tensor is empty");
    
    // TensorShape class  
    py::class_<ma_core::TensorShape>(m, "TensorShape")
        .def_readonly("batch_size", &ma_core::TensorShape::batch_size)
        .def_readonly("sequence_length", &ma_core::TensorShape::sequence_length)
        .def_readonly("num_heads", &ma_core::TensorShape::num_heads)
        .def_readonly("head_dim", &ma_core::TensorShape::head_dim)
        .def("total_elements", &ma_core::TensorShape::total_elements);
    
    // Tensor class (basic interface)
    py::class_<ma_core::Tensor>(m, "Tensor")
        .def("size", &ma_core::Tensor::size, "Total number of elements")
        .def("nbytes", &ma_core::Tensor::nbytes, "Total number of bytes")
        .def("empty", &ma_core::Tensor::empty, "Check if tensor is empty")
        .def("shape", &ma_core::Tensor::shape, "Get tensor shape")
        .def("zero", &ma_core::Tensor::zero, "Fill tensor with zeros")
        .def("fill", &ma_core::Tensor::fill, "Fill tensor with a value")
        // Bulk copy from a contiguous NumPy array (float32)
        .def("copy_from_numpy", [](ma_core::Tensor& t, py::array_t<float, py::array::c_style | py::array::forcecast> arr){
                if (static_cast<size_t>(arr.size()) * sizeof(float) != t.nbytes()) {
                    throw std::runtime_error("copy_from_numpy: size mismatch");
                }
                // Ensure C-contiguous
                auto buf = arr.request();
                t.copy_from(buf.ptr, t.nbytes());
            }, py::arg("array"))
        // Return a new NumPy array with the tensor contents (float32)
        .def("to_numpy", [](const ma_core::Tensor& t){
                const auto& s = t.shape();
                std::array<py::ssize_t,4> shape = {static_cast<py::ssize_t>(s.batch_size),
                                                   static_cast<py::ssize_t>(s.sequence_length),
                                                   static_cast<py::ssize_t>(s.num_heads),
                                                   static_cast<py::ssize_t>(s.head_dim)};
                std::array<py::ssize_t,4> strides = {static_cast<py::ssize_t>(s.sequence_length * s.num_heads * s.head_dim * sizeof(float)),
                                                     static_cast<py::ssize_t>(s.num_heads * s.head_dim * sizeof(float)),
                                                     static_cast<py::ssize_t>(s.head_dim * sizeof(float)),
                                                     static_cast<py::ssize_t>(sizeof(float))};
                // Allocate a new array and copy
                py::array_t<float> out(shape);
                auto buf = out.request();
                t.copy_to(buf.ptr, t.nbytes());
                return out;
            });
    
    // Device enum
    py::enum_<ma_core::Device>(m, "Device")
        .value("CPU", ma_core::Device::CPU)
        .value("MPS", ma_core::Device::MPS)
        .value("CUDA", ma_core::Device::CUDA)
        .value("ROCm", ma_core::Device::ROCm);
}
