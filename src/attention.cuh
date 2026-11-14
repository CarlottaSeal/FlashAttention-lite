#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>
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

    // 求每行最大值
    float mx = -FLT_MAX;
    for (int j = tid; j < N; j += T) mx = fmaxf(mx, row[j]);
    sdata[tid] = mx;
    __syncthreads();
    for (int s = T / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    mx = sdata[0];

    // 减掉最大值再取 exp，顺便求和
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
// K/V load and the Q*K^T reduction. Running (m, l) in shared memory.
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

        // S[b] = (Q * K[b]) * scale. Each warp handles BC/NUM_WARPS rows; within
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

        // sS 这一段的行最大值
        float m_tile = -FLT_MAX;
        for (int b = tid; b < tile_rows; b += T) m_tile = fmaxf(m_tile, sS[b]);
        // 跨 warp 的 shmem 归约：128 线程 -> 1 个值
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

        // P[b] = exp(S[b] - m_new)，顺便累加 p_sum
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

        // 旧 O 乘 alpha 再加这一块的 P*V：O[d] = alpha*O[d] + sum_b P[b]*V[b,d]
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
// block and reused across all Br rows - amortises v1's dominant memory cost.
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

    // 协作加载 Q 的几行，O 清零，初始化 (m, l)
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
            // S[b] = (sQ[warp_id] * sK[b]) * scale, lane-parallel dot product + warp reduction
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

            // 旧 O 乘 alpha 再加这一块的 P*V：O[d] = alpha*O[d] + sum_b P[b]*V[b,d]
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

// v3: FP16 input, Tensor Core matmul via wmma. 16x16 Q/K/V tiles match the
// 16x16x16 fragment shape; one warp per block. Softmax and the running
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

    // 加载 Q tile，初始化 O / m / l
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

        // 加载 K、V tile（FP16）
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

        // S = Q * K^T via wmma, accumulated over D/16 tiles
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

        // 逐行 softmax（FP32），P 转 FP16 写出，旧 O 先乘 alpha
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

        // O += P * V via wmma, one fragment per D tile
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

    // 最后归一化，写回 global memory
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

// v4: BR=64 with 4 warps per block, each warp handles its own 16-row Q slice.
// K/V tile is loaded once and reused by all 4 warps' wmma matmuls, giving the
// Tensor Cores 4x the work per SM compared to v3.
//
// Uses dynamic shared memory - at D=128, BR=64 the tiles need ~63 KB which
// exceeds ptxas's 48 KB static-shmem ceiling. Dynamic shmem with
// cudaFuncAttributeMaxDynamicSharedMemorySize opts into Ada's 100 KB/SM cap.
template <int D_EXACT, int NUM_WARPS>
__global__ void flash_attn_wmma_v4_kernel(const __half* __restrict__ Q,
                                          const __half* __restrict__ K,
                                          const __half* __restrict__ V,
                                          float*        __restrict__ O,
                                          int N, float scale)
{
    using namespace nvcuda::wmma;
    constexpr int WARP_BR = 16;
    constexpr int BR = WARP_BR * NUM_WARPS;
    constexpr int BC = 16;
    constexpr int WM = 16, WN = 16, WK = 16;
    static_assert(D_EXACT % WK == 0, "D must be a multiple of 16");
    constexpr int D_TILES = D_EXACT / WK;
    constexpr int T = NUM_WARPS * 32;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int i0      = blockIdx.x * BR;
    int wr_base = warp_id * WARP_BR;

    extern __shared__ unsigned char smem_raw[];
    float*  sS = reinterpret_cast<float*>(smem_raw);
    float*  sO = sS + BR * BC;
    float*  sm = sO + BR * D_EXACT;
    float*  sl = sm + BR;
    __half* sQ = reinterpret_cast<__half*>(sl + BR);
    __half* sK = sQ + BR * D_EXACT;
    __half* sV = sK + BC * D_EXACT;
    __half* sP = sV + BC * D_EXACT;

    for (int k = tid; k < BR * D_EXACT; k += T) {
        int r = k / D_EXACT, d = k % D_EXACT;
        int row = i0 + r;
        sQ[r * D_EXACT + d] = (row < N) ? Q[row * D_EXACT + d] : __float2half(0.0f);
        sO[r * D_EXACT + d] = 0.0f;
    }
    if (tid < BR) { sm[tid] = -FLT_MAX; sl[tid] = 0.0f; }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BC)
    {
        int tile_rows = min(BC, N - j0);

        for (int k = tid; k < BC * D_EXACT; k += T) {
            int b = k / D_EXACT, d = k % D_EXACT;
            __half kv = (b < tile_rows) ? K[(j0 + b) * D_EXACT + d] : __float2half(0.0f);
            __half vv = (b < tile_rows) ? V[(j0 + b) * D_EXACT + d] : __float2half(0.0f);
            sK[b * D_EXACT + d] = kv;
            sV[b * D_EXACT + d] = vv;
        }
        __syncthreads();

        // Each warp: Q_slice * K^T via wmma
        fragment<accumulator, WM, WN, WK, float> s_frag;
        fill_fragment(s_frag, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < D_TILES; ++kt) {
            fragment<matrix_a, WM, WN, WK, __half, row_major> a_frag;
            fragment<matrix_b, WM, WN, WK, __half, col_major> b_frag;
            load_matrix_sync(a_frag, sQ + wr_base * D_EXACT + kt * WK, D_EXACT);
            load_matrix_sync(b_frag, sK + kt * WK,                     D_EXACT);
            mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        store_matrix_sync(sS + wr_base * BC, s_frag, BC, mem_row_major);
        __syncwarp();

        // 对本 warp 负责的 16 行做 softmax，只有前 16 个 lane 干活，其余空转
        if (lane < WARP_BR) {
            int row = wr_base + lane;
            float* sS_row = sS + row * BC;
            __half* sP_row = sP + row * BC;
            float* sO_row = sO + row * D_EXACT;

            float m_tile = -FLT_MAX;
            for (int j = 0; j < tile_rows; ++j) m_tile = fmaxf(m_tile, sS_row[j] * scale);
            for (int j = tile_rows; j < BC; ++j) sS_row[j] = -FLT_MAX;

            float m_old = sm[row];
            float m_new = fmaxf(m_old, m_tile);
            float alpha = expf(m_old - m_new);

            float p_sum = 0.0f;
            for (int j = 0; j < BC; ++j) {
                float sv = sS_row[j];
                float p  = (sv > -FLT_MAX / 2) ? expf(sv * scale - m_new) : 0.0f;
                sP_row[j] = __float2half(p);
                p_sum += p;
            }
            sl[row] = alpha * sl[row] + p_sum;
            sm[row] = m_new;

            #pragma unroll
            for (int d = 0; d < D_EXACT; ++d) sO_row[d] *= alpha;
        }
        __syncthreads();

        // Each warp: O_slice += P_slice * V via wmma
        #pragma unroll
        for (int kt = 0; kt < D_TILES; ++kt) {
            fragment<accumulator, WM, WN, WK, float>          o_frag;
            fragment<matrix_a, WM, WN, WK, __half, row_major> p_frag;
            fragment<matrix_b, WM, WN, WK, __half, row_major> v_frag;
            load_matrix_sync(o_frag, sO + wr_base * D_EXACT + kt * WK, D_EXACT, mem_row_major);
            load_matrix_sync(p_frag, sP + wr_base * BC,                BC);
            load_matrix_sync(v_frag, sV + kt * WK,                     D_EXACT);
            mma_sync(o_frag, p_frag, v_frag, o_frag);
            store_matrix_sync(sO + wr_base * D_EXACT + kt * WK, o_frag, D_EXACT, mem_row_major);
        }
        __syncthreads();
    }

    for (int k = tid; k < BR * D_EXACT; k += T) {
        int r = k / D_EXACT, d = k % D_EXACT;
        int row = i0 + r;
        if (row < N) {
            float inv_l = 1.0f / sl[r];
            O[row * D_EXACT + d] = sO[r * D_EXACT + d] * inv_l;
        }
    }
}

template <int D_EXACT, int NUM_WARPS>
inline constexpr size_t flash_v4_smem_bytes()
{
    constexpr int BR = 16 * NUM_WARPS;
    constexpr int BC = 16;
    size_t floats  = (size_t)BR * BC + (size_t)BR * D_EXACT + 2 * BR;
    size_t halves  = (size_t)BR * D_EXACT + 2 * (size_t)BC * D_EXACT + (size_t)BR * BC;
    return floats * sizeof(float) + halves * sizeof(__half);
}

inline void run_attention_flash_wmma_v4(const __half* dQ, const __half* dK, const __half* dV,
                                        float* dO, int N, int D, cudaStream_t stream = 0)
{
    constexpr int NUM_WARPS = 4;
    constexpr int BR = 16 * NUM_WARPS;  // 64
    float scale = 1.0f / sqrtf((float)D);
    int grid  = (N + BR - 1) / BR;
    int block = NUM_WARPS * 32;

    if (D == 64) {
        constexpr size_t smem = flash_v4_smem_bytes<64, NUM_WARPS>();
        cudaFuncSetAttribute(flash_attn_wmma_v4_kernel<64, NUM_WARPS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        flash_attn_wmma_v4_kernel<64, NUM_WARPS>
            <<<grid, block, smem, stream>>>(dQ, dK, dV, dO, N, scale);
    } else if (D == 128) {
        constexpr size_t smem = flash_v4_smem_bytes<128, NUM_WARPS>();
        cudaFuncSetAttribute(flash_attn_wmma_v4_kernel<128, NUM_WARPS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        flash_attn_wmma_v4_kernel<128, NUM_WARPS>
            <<<grid, block, smem, stream>>>(dQ, dK, dV, dO, N, scale);
    } else {
        std::fprintf(stderr, "run_attention_flash_wmma_v4: D=%d not in {64, 128}\n", D);
    }
}

// v5: FA-2 structure on raw mma.sync. Q/K/V are contiguous [B, H, N, D];
// blockIdx.y picks the (b, h) slice. Each warp owns 16 query rows. S, P and O
// live in registers; only Q/K/V go through shared memory, XOR-swizzled so that
// ldmatrix reads 8 rows from 8 different 16 B bank groups.
__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const __half* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const __half* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

// c(16x8 f32) += a(16x16 f16, row) * b(16x8 f16, col)
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float x, float y)
{
    __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}

// Row r of a [rows, D] half tile is D/8 chunks of 16 B; chunk c is stored at c ^ (r % 8).
template <int D>
__device__ __forceinline__ int swz(int r, int d)
{
    return r * D + ((((d >> 3) ^ r) & 7) | ((d >> 3) & ~7)) * 8 + (d & 7);
}

template <int D, int BC>
__global__ void __launch_bounds__(128)
flash_attn_v5_kernel(const __half* __restrict__ Q,
                     const __half* __restrict__ K,
                     const __half* __restrict__ V,
                     float*        __restrict__ O,
                     int N, float scale)
{
    constexpr int BR = 64, T = 128;
    constexpr int KD = D / 16;    // k-steps over D for S = QK^T
    constexpr int NS = BC / 8;    // n-tiles of S
    constexpr int NO = D / 8;     // n-tiles of O
    static_assert(D % 16 == 0 && BC % 16 == 0, "");

    size_t off = (size_t)blockIdx.y * N * D;
    Q += off; K += off; V += off; O += off;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);
    __half* sK = sQ + BR * D;
    __half* sV = sK + BC * D;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wr   = (tid >> 5) * 16;   // this warp's first row in the tile
    int g    = lane >> 2;         // accumulator row (and row + 8)
    int t    = lane & 3;          // accumulator column pair
    int i0   = blockIdx.x * BR;

    for (int k = tid; k < BR * D; k += T) {
        int r = k / D, d = k % D;
        sQ[swz<D>(r, d)] = (i0 + r < N) ? Q[(i0 + r) * D + d] : __float2half(0.0f);
    }
    __syncthreads();

    uint32_t qf[KD][4];
    #pragma unroll
    for (int kk = 0; kk < KD; ++kk)
        ldsm_x4(qf[kk], sQ + swz<D>(wr + (lane & 15), kk * 16 + (lane >> 4) * 8));

    float o[NO][4] = {};
    float m[2] = { -CUDART_INF_F, -CUDART_INF_F };
    float l[2] = { 0.0f, 0.0f };   // per-thread partial row sums, reduced at the end

    for (int j0 = 0; j0 < N; j0 += BC)
    {
        __syncthreads();   // everyone is done reading the previous K/V tile
        for (int k = tid; k < BC * D; k += T) {
            int b = k / D, d = k % D;
            bool in = j0 + b < N;
            sK[swz<D>(b, d)] = in ? K[(j0 + b) * D + d] : __float2half(0.0f);
            sV[swz<D>(b, d)] = in ? V[(j0 + b) * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        // S = Q K^T. One ldmatrix.x4 gives the B fragments of two 8-key n-tiles.
        float s[NS][4] = {};
        #pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
            #pragma unroll
            for (int nb = 0; nb < NS / 2; ++nb) {
                uint32_t b[4];
                ldsm_x4(b, sK + swz<D>(nb * 16 + (lane & 7) + (lane >> 4) * 8,
                                       kk * 16 + ((lane >> 3) & 1) * 8));
                mma16816(s[2 * nb],     qf[kk], b[0], b[1]);
                mma16816(s[2 * nb + 1], qf[kk], b[2], b[3]);
            }
        }

        // Online softmax on registers. s[n][0..1] is row g, s[n][2..3] is row g+8;
        // the 4 lanes of a quad hold the rest of each row.
        float mx[2] = { m[0], m[1] };
        #pragma unroll
        for (int n = 0; n < NS; ++n)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int col = j0 + n * 8 + 2 * t + (e & 1);
                s[n][e] = (col < N) ? s[n][e] * scale : -CUDART_INF_F;
                mx[e >> 1] = fmaxf(mx[e >> 1], s[n][e]);
            }
        float alpha[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffff, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffff, mx[i], 2));
            alpha[i] = expf(m[i] - mx[i]);
            m[i] = mx[i];
            l[i] *= alpha[i];
        }
        #pragma unroll
        for (int n = 0; n < NS; ++n)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                s[n][e] = expf(s[n][e] - m[e >> 1]);
                l[e >> 1] += s[n][e];
            }
        #pragma unroll
        for (int n = 0; n < NO; ++n)
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[n][e] *= alpha[e >> 1];

        // O += P V. The S accumulators of n-tiles 2k, 2k+1 are exactly the A
        // fragment for k-step k, so P never leaves registers.
        #pragma unroll
        for (int kk = 0; kk < BC / 16; ++kk) {
            uint32_t a[4] = { pack_half2(s[2 * kk][0],     s[2 * kk][1]),
                              pack_half2(s[2 * kk][2],     s[2 * kk][3]),
                              pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                              pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
            #pragma unroll
            for (int nb = 0; nb < NO / 2; ++nb) {
                uint32_t b[4];
                ldsm_x4_t(b, sV + swz<D>(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                         nb * 16 + (lane >> 4) * 8));
                mma16816(o[2 * nb],     a, b[0], b[1]);
                mma16816(o[2 * nb + 1], a, b[2], b[3]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffff, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffff, l[i], 2);
    }
    int r0 = i0 + wr + g, r1 = r0 + 8;
    #pragma unroll
    for (int n = 0; n < NO; ++n) {
        int c = n * 8 + 2 * t;
        if (r0 < N) *reinterpret_cast<float2*>(O + r0 * D + c) = make_float2(o[n][0] / l[0], o[n][1] / l[0]);
        if (r1 < N) *reinterpret_cast<float2*>(O + r1 * D + c) = make_float2(o[n][2] / l[1], o[n][3] / l[1]);
    }
}

template <int D, int BC>
inline void launch_v5(const __half* dQ, const __half* dK, const __half* dV, float* dO,
                      int BH, int N, cudaStream_t stream)
{
    constexpr int BR = 64;
    constexpr size_t smem = (size_t)(BR + 2 * BC) * D * sizeof(__half);
    cudaFuncSetAttribute(flash_attn_v5_kernel<D, BC>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((N + BR - 1) / BR, BH);
    flash_attn_v5_kernel<D, BC><<<grid, 128, smem, stream>>>(dQ, dK, dV, dO, N,
                                                            1.0f / sqrtf((float)D));
}

template <int D>
inline void launch_v5_bc(const __half* dQ, const __half* dK, const __half* dV, float* dO,
                         int BH, int N, int BC, cudaStream_t stream)
{
    if      (BC == 16) launch_v5<D, 16>(dQ, dK, dV, dO, BH, N, stream);
    else if (BC == 32) launch_v5<D, 32>(dQ, dK, dV, dO, BH, N, stream);
    else               launch_v5<D, 64>(dQ, dK, dV, dO, BH, N, stream);
}

inline void run_attention_flash_v5(const __half* dQ, const __half* dK, const __half* dV,
                                   float* dO, int BH, int N, int D, int BC = 0,
                                   cudaStream_t stream = 0)
{
    if (BC == 0) BC = (D == 64) ? 64 : 32;   // D=128, BC=64 needs 231 regs -> 2 blocks/SM
    if      (D == 64)  launch_v5_bc<64>(dQ, dK, dV, dO, BH, N, BC, stream);
    else if (D == 128) launch_v5_bc<128>(dQ, dK, dV, dO, BH, N, BC, stream);
    else std::fprintf(stderr, "run_attention_flash_v5: D=%d not in {64, 128}\n", D);
}

__global__ void float_to_half_kernel(const float* in, __half* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}
