// CUDA implementation for dense and sparse attention forward/backward paths
// Exposed as a single Torch extension module: sparse_attention_cuda

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <algorithm>

// Simple warp-level reductions for float
__inline__ __device__ float warp_allreduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float warp_allreduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

namespace {

inline int64_t idx4(int64_t b, int64_t i, int64_t h, int64_t d,
                    int64_t B, int64_t S, int64_t H, int64_t D) {
    // Layout: [B, S, H, D] with D contiguous
    return (((b * S + i) * H + h) * D + d);
}

// Optimized dense attention forward: one warp computes one (b,h,i) row.
// Online softmax keeps numerical stability without storing full score matrix.
template <typename T>
__global__ void dense_attention_forward_warp(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int64_t B, int64_t S, int64_t H, int64_t D,
    T scale, bool causal)
{
    // Map grid to (b, h, i)
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t i = blockIdx.x; // one warp per row
    if (i >= S) return;

    // lane id within warp
    const int lane = threadIdx.x & 31;

    // Online softmax variables (shared across warp)
    float m_i = -INFINITY; // running max
    float l_i = 0.f;       // running sum of exp

    // Accumulator for output vector: each lane owns a slice of D: d = lane, lane+32, ...
    for (int64_t d = lane; d < D; d += 32) {
        O[idx4(b, i, h, d, B, S, H, D)] = (T)0;
    }

    const int64_t j_end = causal ? (i + 1) : S;

    // Iterate over keys j
    for (int64_t j = 0; j < j_end; ++j) {
        // Compute dot(Q_i, K_j) across warp
        float dot = 0.f;
        for (int64_t d = lane; d < D; d += 32) {
            dot += (float)Q[idx4(b, i, h, d, B, S, H, D)] * (float)K[idx4(b, j, h, d, B, S, H, D)];
        }
        dot = warp_allreduce_sum(dot);
        dot *= (float)scale;

        // Online softmax update
        float m_new = fmaxf(m_i, dot);
        float l_new = l_i * expf(m_i - m_new) + expf(dot - m_new);

        // Compute coefficients to update accumulated output
        float alpha = (l_i == 0.f) ? 0.f : (l_i * expf(m_i - m_new) / l_new);
        float beta  = expf(dot - m_new) / l_new;

        // Update accumulator O_row = alpha * O_row + beta * V_j
        for (int64_t d = lane; d < D; d += 32) {
            float prev = (float)O[idx4(b, i, h, d, B, S, H, D)];
            float vval = (float)V[idx4(b, j, h, d, B, S, H, D)];
            float out = alpha * prev + beta * vval;
            O[idx4(b, i, h, d, B, S, H, D)] = (T)out;
        }

        m_i = m_new;
        l_i = l_new;
    }
}

template <typename T>
__global__ void dense_attention_backward_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ dO,
    T* __restrict__ dQ,
    T* __restrict__ dK,
    T* __restrict__ dV,
    int64_t B, int64_t S, int64_t H, int64_t D,
    T scale, bool causal)
{
    // One thread computes gradients for a single (b, h, i) row
    int64_t b = blockIdx.z;
    int64_t h = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;

    const int64_t j_limit = causal ? (i + 1) : S;

    // 1) Recompute softmax probabilities P_ij for stability
    // Find max score
    T max_score = -std::numeric_limits<T>::infinity();
    for (int64_t j = 0; j < j_limit; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        s *= scale;
        if (s > max_score) max_score = s;
    }

    // Compute exp(scores - max) and sum
    extern __shared__ unsigned char smem_raw[]; // unused placeholder to keep signature generic
    T sum_exp = 0;
    for (int64_t j = 0; j < j_limit; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        sum_exp += expf((float)(s * scale - max_score));
    }
    if (sum_exp == (T)0) sum_exp = (T)1;

    // 2) Compute dP_ij = dot(dO_i, V_j)
    // and accumulate dV using P_ij
    // Also compute softmax probabilities P_ij
    // Accumulate s_dot = sum_j dP_ij * P_ij for the row
    T s_dot = 0;
    // We will store dQ_i locally to avoid atomics
    // Initialize local dQ_i vector
    // For large D this stack array might be big; compute accum on the fly per k
    // We'll accumulate per-k in a loop after we have dScores.

    // First pass: compute P_ij and dP_ij and s_dot; also accumulate dV
    for (int64_t j = 0; j < j_limit; ++j) {
        // compute score for P_ij again (reuse in loops)
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        T P = expf((float)(s * scale - max_score)) / sum_exp;

        // dP_ij = dot(dO_i, V_j)
        T dP = 0;
        for (int64_t d = 0; d < D; ++d) {
            dP += dO[idx4(b,i,h,d,B,S,H,D)] * V[idx4(b,j,h,d,B,S,H,D)];
        }
        s_dot += dP * P;

        // dV_j += P_ij * dO_i (vector add)
        for (int64_t d = 0; d < D; ++d) {
            atomicAdd(&dV[idx4(b,j,h,d,B,S,H,D)], P * dO[idx4(b,i,h,d,B,S,H,D)]);
        }
    }

    // 3) Compute dScores_ij = (dP_ij - s_dot) * P_ij, then accumulate dQ and dK
    for (int64_t j = 0; j < j_limit; ++j) {
        // recompute P_ij
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        T P = expf((float)(s * scale - max_score)) / sum_exp;

        // recompute dP_ij
        T dP = 0;
        for (int64_t d = 0; d < D; ++d) {
            dP += dO[idx4(b,i,h,d,B,S,H,D)] * V[idx4(b,j,h,d,B,S,H,D)];
        }
        T dScores = (dP - s_dot) * P;

        // dQ_i += (dScores * scale) * K_j
        for (int64_t k = 0; k < D; ++k) {
            // unique writer per (b,i,h,k) in this kernel mapping; no atomic needed
            T val = dScores * scale * K[idx4(b,j,h,k,B,S,H,D)];
            dQ[idx4(b,i,h,k,B,S,H,D)] += val;
        }

        // dK_j += (dScores * scale) * Q_i (requires atomics across i)
        for (int64_t k = 0; k < D; ++k) {
            atomicAdd(&dK[idx4(b,j,h,k,B,S,H,D)], dScores * scale * Q[idx4(b,i,h,k,B,S,H,D)]);
        }
    }
}

// Optimized sparse (sliding window) attention forward: warp-per-row online softmax.
template <typename T>
__global__ void sparse_attention_forward_warp(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int64_t B, int64_t S, int64_t H, int64_t D,
    int64_t window, T scale)
{
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t i = blockIdx.x; // one warp per row
    if (i >= S) return;

    const int lane = threadIdx.x & 31;

    // Initialize output accumulator to zeros for each lane's slice
    for (int64_t d = lane; d < D; d += 32) {
        O[idx4(b, i, h, d, B, S, H, D)] = (T)0;
    }

    const int64_t start_j = max<int64_t>(0, i - window);
    const int64_t end_j   = min<int64_t>(S, i + window + 1);

    float m_i = -INFINITY;
    float l_i = 0.f;

    for (int64_t j = start_j; j < end_j; ++j) {
        float dot = 0.f;
        for (int64_t d = lane; d < D; d += 32) {
            dot += (float)Q[idx4(b, i, h, d, B, S, H, D)] * (float)K[idx4(b, j, h, d, B, S, H, D)];
        }
        dot = warp_allreduce_sum(dot);
        dot *= (float)scale;

        float m_new = fmaxf(m_i, dot);
        float l_new = l_i * expf(m_i - m_new) + expf(dot - m_new);
        float alpha = (l_i == 0.f) ? 0.f : (l_i * expf(m_i - m_new) / l_new);
        float beta  = expf(dot - m_new) / l_new;

        for (int64_t d = lane; d < D; d += 32) {
            float prev = (float)O[idx4(b, i, h, d, B, S, H, D)];
            float vval = (float)V[idx4(b, j, h, d, B, S, H, D)];
            float out = alpha * prev + beta * vval;
            O[idx4(b, i, h, d, B, S, H, D)] = (T)out;
        }

        m_i = m_new;
        l_i = l_new;
    }
}

template <typename T>
__global__ void sparse_attention_backward_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ dO,
    T* __restrict__ dQ,
    T* __restrict__ dK,
    T* __restrict__ dV,
    int64_t B, int64_t S, int64_t H, int64_t D,
    int64_t window, T scale)
{
    // One thread computes gradients for a single (b, h, i)
    int64_t b = blockIdx.z;
    int64_t h = blockIdx.y;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;

    int64_t start_j = max<int64_t>(0, i - window);
    int64_t end_j = min<int64_t>(S, i + window + 1);

    // 1) Recompute softmax probs within window
    T max_score = -std::numeric_limits<T>::infinity();
    for (int64_t j = start_j; j < end_j; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        s *= scale;
        if (s > max_score) max_score = s;
    }
    T sum_exp = 0;
    for (int64_t j = start_j; j < end_j; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        sum_exp += expf((float)(s * scale - max_score));
    }
    if (sum_exp == (T)0) sum_exp = (T)1;

    // 2) First pass: compute dP and s_dot; accumulate dV
    T s_dot = 0;
    for (int64_t j = start_j; j < end_j; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        T P = expf((float)(s * scale - max_score)) / sum_exp;

        T dP = 0;
        for (int64_t d = 0; d < D; ++d) {
            dP += dO[idx4(b,i,h,d,B,S,H,D)] * V[idx4(b,j,h,d,B,S,H,D)];
        }
        s_dot += dP * P;

        for (int64_t d = 0; d < D; ++d) {
            atomicAdd(&dV[idx4(b,j,h,d,B,S,H,D)], P * dO[idx4(b,i,h,d,B,S,H,D)]);
        }
    }

    // 3) Second pass: compute dScores, accumulate dQ and dK
    for (int64_t j = start_j; j < end_j; ++j) {
        T s = 0;
        for (int64_t d = 0; d < D; ++d) {
            s += Q[idx4(b,i,h,d,B,S,H,D)] * K[idx4(b,j,h,d,B,S,H,D)];
        }
        T P = expf((float)(s * scale - max_score)) / sum_exp;

        T dP = 0;
        for (int64_t d = 0; d < D; ++d) {
            dP += dO[idx4(b,i,h,d,B,S,H,D)] * V[idx4(b,j,h,d,B,S,H,D)];
        }
        T dScores = (dP - s_dot) * P;

        for (int64_t k = 0; k < D; ++k) {
            // unique writer for dQ element; atomics still required for dK across i
            dQ[idx4(b,i,h,k,B,S,H,D)] += dScores * scale * K[idx4(b,j,h,k,B,S,H,D)];
            atomicAdd(&dK[idx4(b,j,h,k,B,S,H,D)], dScores * scale * Q[idx4(b,i,h,k,B,S,H,D)]);
        }
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

    // One warp (32 threads) per (b,h,i) row for good parallelism and coalesced access.
    dim3 block(32);
    dim3 grid( static_cast<unsigned int>(S),
               static_cast<unsigned int>(H),
               static_cast<unsigned int>(B) );
    auto stream = at::cuda::getCurrentCUDAStream();

    dense_attention_forward_warp<float><<<grid, block, 0, stream.stream()>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        out.data_ptr<float>(), B, S, H, D, scale, use_causal_mask);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> dense_backward_cuda(torch::Tensor query,
                                               torch::Tensor key,
                                               torch::Tensor value,
                                               torch::Tensor grad_out,
                                               bool use_causal_mask) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && grad_out.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.dtype() == torch::kFloat32 && key.dtype() == torch::kFloat32 && value.dtype() == torch::kFloat32 && grad_out.dtype() == torch::kFloat32,
                "Only float32 is supported in CUDA path");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() && grad_out.is_contiguous(), "Inputs must be contiguous" );
    TORCH_CHECK(query.sizes() == key.sizes() && key.sizes() == value.sizes() && value.sizes() == grad_out.sizes(), "All tensors must have same shape");
    TORCH_CHECK(query.dim() == 4, "Expected shape [B, S, H, D]");

    auto B = query.size(0);
    auto S = query.size(1);
    auto H = query.size(2);
    auto D = query.size(3);

    auto options = query.options();
    auto dQ = torch::zeros({B, S, H, D}, options);
    auto dK = torch::zeros({B, S, H, D}, options);
    auto dV = torch::zeros({B, S, H, D}, options);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 block(32);
    dim3 grid( static_cast<unsigned int>(S),
               static_cast<unsigned int>(H),
               static_cast<unsigned int>(B) );
    auto stream = at::cuda::getCurrentCUDAStream();

    dense_attention_backward_kernel<float><<<grid, block, 0, stream.stream()>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(), grad_out.data_ptr<float>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        B, S, H, D, scale, use_causal_mask);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dQ, dK, dV};
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

    sparse_attention_forward_warp<float><<<grid, block, 0, stream.stream()>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        out.data_ptr<float>(), B, S, H, D, window_size, scale);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> sparse_backward_cuda(torch::Tensor query,
                                                torch::Tensor key,
                                                torch::Tensor value,
                                                torch::Tensor grad_out,
                                                int64_t window_size) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && grad_out.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.dtype() == torch::kFloat32 && key.dtype() == torch::kFloat32 && value.dtype() == torch::kFloat32 && grad_out.dtype() == torch::kFloat32,
                "Only float32 is supported in CUDA path");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() && grad_out.is_contiguous(), "Inputs must be contiguous" );
    TORCH_CHECK(query.sizes() == key.sizes() && key.sizes() == value.sizes() && value.sizes() == grad_out.sizes(), "All tensors must have same shape");
    TORCH_CHECK(query.dim() == 4, "Expected shape [B, S, H, D]");

    auto B = query.size(0);
    auto S = query.size(1);
    auto H = query.size(2);
    auto D = query.size(3);

    auto options = query.options();
    auto dQ = torch::zeros({B, S, H, D}, options);
    auto dK = torch::zeros({B, S, H, D}, options);
    auto dV = torch::zeros({B, S, H, D}, options);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    unsigned int block_x = static_cast<unsigned int>(std::min<int64_t>(S, 128));
    dim3 block(block_x);
    dim3 grid( static_cast<unsigned int>((S + block.x - 1) / block.x),
               static_cast<unsigned int>(H),
               static_cast<unsigned int>(B) );
    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_attention_backward_kernel<float><<<grid, block, 0, stream.stream()>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(), grad_out.data_ptr<float>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(), B, S, H, D, window_size, scale);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dQ, dK, dV};
}

} // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_forward", &dense_forward_cuda, "Dense attention forward (CUDA)",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("use_causal_mask") = false);
    m.def("forward", &sparse_forward_cuda, "Sparse attention forward (CUDA)",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("window_size") = 64);
    m.def("dense_backward", &dense_backward_cuda, "Dense attention backward (CUDA)",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("grad_out"), py::arg("use_causal_mask") = false);
    m.def("backward", &sparse_backward_cuda, "Sparse attention backward (CUDA)",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("grad_out"), py::arg("window_size") = 64);
}
