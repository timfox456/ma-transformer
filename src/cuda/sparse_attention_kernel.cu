
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor sparse_attention_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value);

// C++ interface
torch::Tensor sparse_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
    return sparse_attention_forward_cuda(query, key, value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_attention_forward, "Sparse Attention forward (CUDA)");
}
