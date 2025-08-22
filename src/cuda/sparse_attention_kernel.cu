// CUDA implementation for dense and sparse attention forward paths
// Exposed as a single Torch extension module: sparse_attention_cuda

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <algorithm>

namespace {

inline int64_t idx4(int64_t b, int64_t i, int64_t h, int64_t d,
                    int64_t B, int64_t S, int64_t H, int64_t D) {
    // Layout: [B, S, H, D] with D contiguous
    return (((b * S + i) * H + h) * D + d);
}

template <typename T>
__global__ void dense_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int64_t B, int64_t S, int64_t H, int64_t D,
    T scale, bool causal)
{
    // One thread computes one (b, h, i) row of attention
    int64_t b = blockIdx.z;
    int64_t h = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;

    // 1) find max score over j
    T max_score = -std::numeric_limits<T>::infinity();
    for (int64_t j = 0; j < S; ++j) {
        if (causal && j > i) break; // scores after i are masked
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        s *= scale;
        if (s > max_score) max_score = s;
    }

    // 2) compute sum_exp
    T sum_exp = 0;
    int64_t j_limit = causal ? (i + 1) : S;
    for (int64_t j = 0; j < j_limit; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        s = expf((float)(s * scale - max_score));
        sum_exp += s;
    }
    if (sum_exp == (T)0) sum_exp = (T)1; // guard against division by zero

    // 3) accumulate output = softmax(scores) @ V
    for (int64_t d = 0; d < D; ++d) {
        T acc = 0;
        for (int64_t j = 0; j < j_limit; ++j) {
            T s = 0;
            for (int64_t k = 0; k < D; ++k) {
                s += Q[idx4(b,i,h,k,B,S,H,D)] * K[idx4(b,j,h,k,B,S,H,D)];
            }
            s = expf((float)(s * scale - max_score)) / sum_exp;
            acc += s * V[idx4(b,j,h,d,B,S,H,D)];
        }
        O[idx4(b,i,h,d,B,S,H,D)] = acc;
    }
}

template <typename T>
__global__ void sparse_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int64_t B, int64_t S, int64_t H, int64_t D,
    int64_t window, T scale)
{
    // One thread computes one (b, h, i)
    int64_t b = blockIdx.z;
    int64_t h = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;

    int64_t start_j = max<int64_t>(0, i - window);
    int64_t end_j = min<int64_t>(S, i + window + 1);

    // 1) find max
    T max_score = -std::numeric_limits<T>::infinity();
    for (int64_t j = start_j; j < end_j; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        s *= scale;
        if (s > max_score) max_score = s;
    }

    // 2) sum_exp
    T sum_exp = 0;
    for (int64_t j = start_j; j < end_j; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        s = expf((float)(s * scale - max_score));
        sum_exp += s;
    }
    if (sum_exp == (T)0) sum_exp = (T)1;

    // 3) output
    for (int64_t d = 0; d < D; ++d) {
        T acc = 0;
        for (int64_t j = start_j; j < end_j; ++j) {
            T s = 0;
            for (int64_t k = 0; k < D; ++k) {
                s += Q[idx4(b,i,h,k,B,S,H,D)] * K[idx4(b,j,h,k,B,S,H,D)];
            }
            s = expf((float)(s * scale - max_score)) / sum_exp;
            acc += s * V[idx4(b,j,h,d,B,S,H,D)];
        }
        O[idx4(b,i,h,d,B,S,H,D)] = acc;
    }
}

torch::Tensor dense_forward_cuda(torch::Tensor query,
                                 torch::Tensor key,
                                 torch::Tensor value,
                                 bool use_causal_mask) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.dtype() == torch::kFloat32 && key.dtype() == torch::kFloat32 && value.dtype() == torch::kFloat32,
                "Only float32 is supported in CUDA path");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(), "Inputs must be contiguous" );
    TORCH_CHECK(query.sizes() == key.sizes() && key.sizes() == value.sizes(), "Q,K,V must have same shape");
    TORCH_CHECK(query.dim() == 4, "Expected shape [B, S, H, D]");

    auto B = query.size(0);
    auto S = query.size(1);
    auto H = query.size(2);
    auto D = query.size(3);

    auto options = query.options();
    auto out = torch::empty({B, S, H, D}, options);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    unsigned int block_x = static_cast<unsigned int>(std::min<int64_t>(S, 128));
    dim3 block(block_x);
    dim3 grid( static_cast<unsigned int>((S + block.x - 1) / block.x),
               static_cast<unsigned int>(H),
               static_cast<unsigned int>(B) );
    auto stream = at::cuda::getCurrentCUDAStream();

    dense_attention_kernel<float><<<grid, block, 0, stream.stream()>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        out.data_ptr<float>(), B, S, H, D, scale, use_causal_mask);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor sparse_forward_cuda(torch::Tensor query,
                                  torch::Tensor key,
                                  torch::Tensor value,
                                  int64_t window_size) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.dtype() == torch::kFloat32 && key.dtype() == torch::kFloat32 && value.dtype() == torch::kFloat32,
                "Only float32 is supported in CUDA path");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(), "Inputs must be contiguous" );
    TORCH_CHECK(query.sizes() == key.sizes() && key.sizes() == value.sizes(), "Q,K,V must have same shape");
    TORCH_CHECK(query.dim() == 4, "Expected shape [B, S, H, D]");

    auto B = query.size(0);
    auto S = query.size(1);
    auto H = query.size(2);
    auto D = query.size(3);

    auto options = query.options();
    auto out = torch::empty({B, S, H, D}, options);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    unsigned int block_x = static_cast<unsigned int>(std::min<int64_t>(S, 128));
    dim3 block(block_x);
    dim3 grid( static_cast<unsigned int>((S + block.x - 1) / block.x),
               static_cast<unsigned int>(H),
               static_cast<unsigned int>(B) );
    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_attention_kernel<float><<<grid, block, 0, stream.stream()>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        out.data_ptr<float>(), B, S, H, D, window_size, scale);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_forward", &dense_forward_cuda, "Dense attention forward (CUDA)",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("use_causal_mask") = false);
    m.def("forward", &sparse_forward_cuda, "Sparse attention forward (CUDA)",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("window_size") = 64);
}
