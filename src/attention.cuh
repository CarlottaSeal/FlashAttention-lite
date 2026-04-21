#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// All kernels: single-head FP32 attention, non-causal.
// Shapes (row-major):
//   Q, K, V, O : [N, D]
//   scratch    : used by kernels that materialize S[N,N]; caller owns it

// ---------- naive: materializes S[N,N] in global memory ----------
__global__ void attn_naive_score_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        float* __restrict__ S,
                                        int N, int D, float scale)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;

    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += Q[i * D + d] * K[j * D + d];
    S[i * N + j] = s * scale;
}

__global__ void attn_naive_softmax_kernel(float* __restrict__ S, int N)
{
    int i = blockIdx.x;
    int tid = threadIdx.x;
    int T = blockDim.x;
    float* row = S + (size_t)i * N;

    extern __shared__ float sdata[];

    // max
    float mx = -FLT_MAX;
    for (int j = tid; j < N; j += T) mx = fmaxf(mx, row[j]);
    sdata[tid] = mx;
    __syncthreads();
    for (int s = T / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    mx = sdata[0];

    // exp + sum
    float sum = 0.0f;
    for (int j = tid; j < N; j += T) {
        float e = expf(row[j] - mx);
        row[j] = e;
        sum += e;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = T / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int j = tid; j < N; j += T) row[j] *= inv_sum;
}

__global__ void attn_naive_out_kernel(const float* __restrict__ P,
                                      const float* __restrict__ V,
                                      float* __restrict__ O,
                                      int N, int D)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || d >= D) return;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j) acc += P[i * N + j] * V[j * D + d];
    O[i * D + d] = acc;
}

inline void run_attention_naive(const float* dQ, const float* dK, const float* dV,
                                float* dO, float* dS, int N, int D, cudaStream_t stream = 0)
{
    float scale = 1.0f / sqrtf((float)D);

    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        attn_naive_score_kernel<<<grid, block, 0, stream>>>(dQ, dK, dS, N, D, scale);
    }
    {
        int T = 128;
        attn_naive_softmax_kernel<<<N, T, T * sizeof(float), stream>>>(dS, N);
    }
    {
        dim3 block(16, 16);
        dim3 grid((D + 15) / 16, (N + 15) / 16);
        attn_naive_out_kernel<<<grid, block, 0, stream>>>(dS, dV, dO, N, D);
    }
}

// ---------- FlashAttention: online softmax, no N×N materialised ----------
// One block per output row, 128 threads = 4 warps. Block-level cooperation
// on K/V load, on S-tile compute (warp-reduced dot product per row), and on
// the O-tile update. Running (m, l) in shared memory.
template <int BC, int D_MAX>
__global__ void flash_attn_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ O,
                                  int N, int D, float scale)
{
    int i   = blockIdx.x;
    int tid = threadIdx.x;           // 0..127
    int warp_id = tid >> 5;          // 0..3
    int lane    = tid & 31;
    const int NUM_WARPS = 4;
    const int T = 128;

    __shared__ float sQ[D_MAX];
    __shared__ float sO[D_MAX];
    __shared__ float sK[BC * D_MAX];
    __shared__ float sV[BC * D_MAX];
    __shared__ float sS[BC];
    __shared__ float s_m, s_l;

    for (int d = tid; d < D; d += T) {
        sQ[d] = Q[i * D + d];
        sO[d] = 0.0f;
    }
    if (tid == 0) { s_m = -FLT_MAX; s_l = 0.0f; }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BC)
    {
        int tile_rows = min(BC, N - j0);

        for (int k = tid; k < tile_rows * D; k += T) {
            int b = k / D, d = k % D;
            sK[b * D + d] = K[(j0 + b) * D + d];
            sV[b * D + d] = V[(j0 + b) * D + d];
        }
        __syncthreads();

        // S[b] = (Q · K[b]) * scale. Each warp handles BC/NUM_WARPS rows; within
        // a warp, 32 lanes cooperatively compute one dot product and warp-reduce.
        for (int b = warp_id; b < tile_rows; b += NUM_WARPS) {
            float partial = 0.0f;
            for (int d = lane; d < D; d += 32) partial += sQ[d] * sK[b * D + d];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            if (lane == 0) sS[b] = partial * scale;
        }
        __syncthreads();

        // Row max of sS
        float m_tile = -FLT_MAX;
        for (int b = tid; b < tile_rows; b += T) m_tile = fmaxf(m_tile, sS[b]);
        // Cross-block reduction through shmem (128 threads → 1 value)
        __shared__ float red[T / 32];   // one slot per warp
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));
        if (lane == 0) red[warp_id] = m_tile;
        __syncthreads();
        if (warp_id == 0) {
            float v = lane < NUM_WARPS ? red[lane] : -FLT_MAX;
            #pragma unroll
            for (int off = 2; off > 0; off >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
            if (lane == 0) red[0] = v;
        }
        __syncthreads();
        float m_old = s_m;
        float m_new = fmaxf(m_old, red[0]);
        float alpha = expf(m_old - m_new);

        // P[b] = exp(S[b] - m_new), accumulate p_sum
        float p_sum = 0.0f;
        for (int b = tid; b < tile_rows; b += T) {
            float p = expf(sS[b] - m_new);
            sS[b] = p;
            p_sum += p;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            p_sum += __shfl_xor_sync(0xffffffff, p_sum, off);
        if (lane == 0) red[warp_id] = p_sum;
        __syncthreads();
        float p_sum_total = 0.0f;
        if (warp_id == 0) {
            float v = lane < NUM_WARPS ? red[lane] : 0.0f;
            #pragma unroll
            for (int off = 2; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0) red[0] = v;
        }
        __syncthreads();
        p_sum_total = red[0];

        if (tid == 0) {
            s_l = alpha * s_l + p_sum_total;
            s_m = m_new;
        }
        __syncthreads();

        // O[d] = alpha * O[d] + sum_b P[b] * V[b, d]
        for (int d = tid; d < D; d += T) {
            float o = alpha * sO[d];
            for (int b = 0; b < tile_rows; ++b) o += sS[b] * sV[b * D + d];
            sO[d] = o;
        }
        __syncthreads();
    }

    float inv_l = 1.0f / s_l;
    for (int d = tid; d < D; d += T) O[i * D + d] = sO[d] * inv_l;
}

inline void run_attention_flash(const float* dQ, const float* dK, const float* dV,
                                float* dO, int N, int D, cudaStream_t stream = 0)
{
    float scale = 1.0f / sqrtf((float)D);
    constexpr int BC = 32;
    constexpr int D_MAX = 128;
    flash_attn_kernel<BC, D_MAX><<<N, 128, 0, stream>>>(dQ, dK, dV, dO, N, D, scale);
}
