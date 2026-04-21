#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cfloat>

// Single-head, non-causal attention. Q, K, V, O are row-major [N, D].
// Naive kernels need an S[N,N] scratch buffer; FlashAttention variants don't.

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

// v1: online softmax, one output row per block, 4 warps cooperating on
// K/V load and the Q·Kᵀ reduction. Running (m, l) in shared memory.
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

// v2: Br rows of Q per block, 1 warp per row. K/V tile loaded once per
// block and reused across all Br rows — amortises v1's dominant memory cost.
template <int BR, int BC, int D_EXACT>
__global__ __launch_bounds__(BR * 32, 2)
void flash_attn_qtile_kernel(const float* __restrict__ Q,
                             const float* __restrict__ K,
                             const float* __restrict__ V,
                             float* __restrict__ O,
                             int N, float scale)
{
    int row_base = blockIdx.x * BR;
    int tid      = threadIdx.x;
    int warp_id  = tid >> 5;
    int lane     = tid & 31;
    int i        = row_base + warp_id;

    __shared__ float sQ[BR][D_EXACT];
    __shared__ float sO[BR][D_EXACT];
    __shared__ float sK[BC][D_EXACT];
    __shared__ float sV[BC][D_EXACT];
    __shared__ float sS[BR][BC];
    __shared__ float sm[BR];
    __shared__ float sl[BR];

    // Cooperatively load Q rows, zero O, init (m, l)
    for (int k = tid; k < BR * D_EXACT; k += BR * 32) {
        int r = k / D_EXACT, d = k % D_EXACT;
        int row_glob = row_base + r;
        sQ[r][d] = (row_glob < N) ? Q[row_glob * D_EXACT + d] : 0.0f;
        sO[r][d] = 0.0f;
    }
    if (tid < BR) { sm[tid] = -FLT_MAX; sl[tid] = 0.0f; }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BC)
    {
        int tile_rows = min(BC, N - j0);

        for (int k = tid; k < tile_rows * D_EXACT; k += BR * 32) {
            int b = k / D_EXACT, d = k % D_EXACT;
            sK[b][d] = K[(j0 + b) * D_EXACT + d];
            sV[b][d] = V[(j0 + b) * D_EXACT + d];
        }
        __syncthreads();

        if (i < N)
        {
            // S[b] = (sQ[warp_id] · sK[b]) * scale, lane-parallel dot product + warp reduction
            for (int b = lane; b < tile_rows; b += 32) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < D_EXACT; ++d) s += sQ[warp_id][d] * sK[b][d];
                sS[warp_id][b] = s * scale;
            }

            // Row max (warp-local, across lanes that touched different b's)
            float m_tile = -FLT_MAX;
            for (int b = lane; b < tile_rows; b += 32) m_tile = fmaxf(m_tile, sS[warp_id][b]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));

            float m_old = sm[warp_id];
            float m_new = fmaxf(m_old, m_tile);
            float alpha = expf(m_old - m_new);

            float p_sum = 0.0f;
            for (int b = lane; b < tile_rows; b += 32) {
                float p = expf(sS[warp_id][b] - m_new);
                sS[warp_id][b] = p;
                p_sum += p;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                p_sum += __shfl_xor_sync(0xffffffff, p_sum, off);

            if (lane == 0) {
                sl[warp_id] = alpha * sl[warp_id] + p_sum;
                sm[warp_id] = m_new;
            }

            // O[d] = alpha * O[d] + sum_b P[b] * V[b, d]
            for (int d = lane; d < D_EXACT; d += 32) {
                float o = alpha * sO[warp_id][d];
                for (int b = 0; b < tile_rows; ++b) o += sS[warp_id][b] * sV[b][d];
                sO[warp_id][d] = o;
            }
        }
        __syncthreads();
    }

    if (i < N) {
        float inv_l = 1.0f / sl[warp_id];
        for (int d = lane; d < D_EXACT; d += 32)
            O[i * D_EXACT + d] = sO[warp_id][d] * inv_l;
    }
}

inline void run_attention_flash_qtile(const float* dQ, const float* dK, const float* dV,
                                      float* dO, int N, int D, cudaStream_t stream = 0)
{
    float scale = 1.0f / sqrtf((float)D);
    constexpr int BR = 4;
    constexpr int BC = 32;
    int grid = (N + BR - 1) / BR;
    int block = BR * 32;   // 1 warp per Q row

    if (D == 64)
        flash_attn_qtile_kernel<BR, BC, 64><<<grid, block, 0, stream>>>(dQ, dK, dV, dO, N, scale);
    else if (D == 128)
        flash_attn_qtile_kernel<BR, BC, 128><<<grid, block, 0, stream>>>(dQ, dK, dV, dO, N, scale);
    else
        std::fprintf(stderr, "run_attention_flash_qtile: D=%d not in {64, 128}\n", D);
}

// v3: FP16 input, Tensor Core matmul via wmma. 16×16 Q/K/V tiles match the
// 16×16×16 fragment shape; one warp per block. Softmax and the running
// (m, l, O) state stay FP32 for stability.
template <int D_EXACT>
__global__ void flash_attn_wmma_kernel(const __half* __restrict__ Q,
                                       const __half* __restrict__ K,
                                       const __half* __restrict__ V,
                                       float*        __restrict__ O,
                                       int N, float scale)
{
    using namespace nvcuda::wmma;
    constexpr int BR = 16;
    constexpr int BC = 16;
    constexpr int WM = 16, WN = 16, WK = 16;
    static_assert(D_EXACT % WK == 0, "D must be a multiple of 16");
    constexpr int D_TILES = D_EXACT / WK;

    int tid = threadIdx.x;        // 0..31, one warp per block
    int i0  = blockIdx.x * BR;    // Q rows handled by this block

    __shared__ __half sQ[BR][D_EXACT];
    __shared__ __half sK[BC][D_EXACT];
    __shared__ __half sV[BC][D_EXACT];
    __shared__ float  sS[BR][BC];       // FP32 post-QK, pre-softmax
    __shared__ __half sP[BR][BC];       // FP16 softmax output, PV input
    __shared__ float  sO[BR][D_EXACT];  // FP32 running accumulator
    __shared__ float  sm[BR];
    __shared__ float  sl[BR];

    // Load Q tile, init O / m / l
    for (int k = tid; k < BR * D_EXACT; k += 32) {
        int r = k / D_EXACT, d = k % D_EXACT;
        int row = i0 + r;
        sQ[r][d] = (row < N) ? Q[row * D_EXACT + d] : __float2half(0.0f);
    }
    for (int k = tid; k < BR * D_EXACT; k += 32) {
        int r = k / D_EXACT, d = k % D_EXACT;
        sO[r][d] = 0.0f;
    }
    if (tid < BR) { sm[tid] = -FLT_MAX; sl[tid] = 0.0f; }
    __syncwarp();

    for (int j0 = 0; j0 < N; j0 += BC)
    {
        int tile_rows = min(BC, N - j0);

        // Load K, V tiles (FP16)
        for (int k = tid; k < BC * D_EXACT; k += 32) {
            int b = k / D_EXACT, d = k % D_EXACT;
            if (b < tile_rows) {
                sK[b][d] = K[(j0 + b) * D_EXACT + d];
                sV[b][d] = V[(j0 + b) * D_EXACT + d];
            } else {
                sK[b][d] = __float2half(0.0f);
                sV[b][d] = __float2half(0.0f);
            }
        }
        __syncwarp();

        // S = Q · Kᵀ via wmma, accumulated over D/16 tiles
        fragment<accumulator, WM, WN, WK, float> s_frag;
        fill_fragment(s_frag, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < D_TILES; ++kt) {
            fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
            fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag;
            load_matrix_sync(a_frag, &sQ[0][kt * WK], D_EXACT);
            load_matrix_sync(b_frag, &sK[0][kt * WK], D_EXACT);
            mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        store_matrix_sync(&sS[0][0], s_frag, BC, mem_row_major);
        __syncwarp();

        // Per-row softmax (FP32), write P as FP16, scale stale O by alpha
        if (tid < BR) {
            int row = tid;
            float m_tile = -FLT_MAX;
            for (int j = 0; j < tile_rows; ++j)
                m_tile = fmaxf(m_tile, sS[row][j] * scale);
            for (int j = tile_rows; j < BC; ++j) sS[row][j] = -FLT_MAX;   // mask padding
            float m_old = sm[row];
            float m_new = fmaxf(m_old, m_tile);
            float alpha = expf(m_old - m_new);

            float p_sum = 0.0f;
            for (int j = 0; j < BC; ++j) {
                float sv = sS[row][j];
                float p  = (sv > -FLT_MAX / 2) ? expf(sv * scale - m_new) : 0.0f;
                sP[row][j] = __float2half(p);
                p_sum += p;
            }
            sl[row] = alpha * sl[row] + p_sum;
            sm[row] = m_new;

            #pragma unroll
            for (int d = 0; d < D_EXACT; ++d) sO[row][d] *= alpha;
        }
        __syncwarp();

        // O += P · V via wmma, one fragment per D tile
        #pragma unroll
        for (int kt = 0; kt < D_TILES; ++kt) {
            fragment<accumulator, WM, WN, WK, float>                o_frag;
            fragment<matrix_a, WM, WN, WK, __half, row_major>       p_frag;
            fragment<matrix_b, WM, WN, WK, __half, row_major>       v_frag;

            load_matrix_sync(o_frag, &sO[0][kt * WK], D_EXACT, mem_row_major);
            load_matrix_sync(p_frag, &sP[0][0],       BC);
            load_matrix_sync(v_frag, &sV[0][kt * WK], D_EXACT);
            mma_sync(o_frag, p_frag, v_frag, o_frag);
            store_matrix_sync(&sO[0][kt * WK], o_frag, D_EXACT, mem_row_major);
        }
        __syncwarp();
    }

    // Final normalize + write to global
    if (tid < BR) {
        int row = tid;
        if (i0 + row < N) {
            float inv_l = 1.0f / sl[row];
            for (int d = 0; d < D_EXACT; ++d)
                O[(i0 + row) * D_EXACT + d] = sO[row][d] * inv_l;
        }
    }
}

inline void run_attention_flash_wmma(const __half* dQ, const __half* dK, const __half* dV,
                                     float* dO, int N, int D, cudaStream_t stream = 0)
{
    float scale = 1.0f / sqrtf((float)D);
    int grid = (N + 15) / 16;
    int block = 32;   // one warp per block
    if (D == 64)
        flash_attn_wmma_kernel<64><<<grid, block, 0, stream>>>(dQ, dK, dV, dO, N, scale);
    else if (D == 128)
        flash_attn_wmma_kernel<128><<<grid, block, 0, stream>>>(dQ, dK, dV, dO, N, scale);
    else
        std::fprintf(stderr, "run_attention_flash_wmma: D=%d not in {64, 128}\n", D);
}

__global__ void float_to_half_kernel(const float* in, __half* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}
