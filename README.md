# FlashAttention-lite

Educational FP32 CUDA implementation of the FlashAttention forward pass.
One-block-per-output-row, online softmax so the N×N attention matrix is
never materialised. Written to understand the algorithm and to benchmark
it against a straightforward multi-kernel baseline and PyTorch's SDPA
(which picks FlashAttention-2 + Tensor Cores under the hood on Ada).

## Algorithm

Inner loop over K/V tiles, per output row:

```
m_new  = max(m_old, max(S_tile))
α      = exp(m_old - m_new)
l_new  = α · l_old + sum(exp(S_tile - m_new))
O_new  = α · O_old + exp(S_tile - m_new) · V_tile
```

After the final tile: `O = O_final / l_final`.

## Correctness

Max absolute difference against a double-precision CPU reference,
across `N ∈ {512, 1024, 2048, 4096}` and `D ∈ {64, 128}`:
**< 2.4e-8 for every case**. Within single-precision noise.

## Benchmark (RTX 4080 Laptop, Ada SM 8.9, FP32)

| Shape | Naive (3-kernel) | **Flash (online softmax)** | PyTorch SDPA FP32 | PyTorch SDPA FP16 |
|---|--:|--:|--:|--:|
| N=512  D=64  | 0.12 ms | 0.33 ms | 0.10 ms | 0.04 ms |
| N=1024 D=64  | 0.42 ms | 1.23 ms | 0.20 ms | 0.06 ms |
| N=2048 D=64  | 1.59 ms | 4.71 ms | 0.39 ms | 0.11 ms |
| N=4096 D=64  | 6.12 ms | 16.4 ms | 1.18 ms | 0.28 ms |
| N=4096 D=128 | 12.8 ms | 24.1 ms | 2.12 ms | 0.43 ms |

My flash kernel is slower than my own multi-kernel naive at this scale.
Nsight Compute breakdown for the flash kernel (`N=2048, D=64`):

| Metric | Value |
|---|--:|
| **Theoretical occupancy** | 16.67% (8 warps / SM) |
| Block limit — shmem | 2 blocks / SM |
| Block limit — registers | 12 blocks / SM (not the limiter) |
| Compute (SM) Throughput | 36.6% |
| L1 / L2 Cache Throughput | 37.4% / 7.9% |
| DRAM Throughput | 1.1% |
| Registers / thread | 40 |

## What's limiting this impl

Three factors explain most of the gap against PyTorch / the naive baseline:

1. **Shared memory dominates occupancy.** Each block holds `sK + sV`
   tiles sized `BC × D_MAX` = 32 × 128 × 4 bytes × 2 = 32 KB. That caps
   the SM at 2 blocks × 4 warps = 8 warps, i.e. 16.7% theoretical
   occupancy. The natural fix is a Q-tile-per-block layout so more rows
   are handled per K/V load, amortising the tile cost and shrinking
   relative shmem usage.
2. **No Tensor Cores.** PyTorch's SDPA in FP16 is ~5× faster than its
   FP32 path on the same hardware because Ada's 4th-gen Tensor Cores do
   the Q·Kᵀ and P·V matmuls. The gap between my FP32 flash and PyTorch
   FP16 SDPA (16.4 ms vs 0.28 ms at N=4096 D=64) is almost entirely
   FP16 + Tensor Core.
3. **One row per block.** Real FlashAttention tiles Q across blocks,
   keeping multiple output rows' `(m, l, O)` state together. Amortises
   K/V loads across the Q tile — the main arithmetic intensity win.

None of these three invalidate the algorithm — which is demonstrably
correct at 1e-8 precision. They're the engineering steps between
"educational implementation" and "production kernel".

## Build & run

```bat
build.bat
build\bin\attn.exe --N 2048 --D 64
```

The binary runs naive + flash, checks both against a CPU double-precision
reference, and prints per-kernel timings.

PyTorch reference numbers:
```bash
python scripts\torch_bench.py
```

## Files

```
src/
  attention.cuh   naive kernels + flash kernel
  main.cu         test harness: CPU ref, correctness diff, timing
scripts/
  torch_bench.py  PyTorch SDPA reference timings
docs/
  profile/flash.ncu-rep   raw Nsight Compute report
build.bat
```

## Hardware

RTX 4080 Laptop (Ada Lovelace, SM 8.9, 58 SMs, 12 GB, ~40 MB L2, 100 KB shmem/SM).
