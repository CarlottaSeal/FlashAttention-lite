# FlashAttention-lite

FP32 CUDA FlashAttention forward, written to understand the algorithm
and to land on a kernel that actually beats a naive multi-kernel
baseline. Two flavours here:

- **v1 (didactic):** 1 output row per block, 1 warp. Minimum-viable
  online softmax. Slower than naive, useful as a correctness reference.
- **v2 (Q-tile):** 4 rows per block, 1 warp per row, K/V tile shared
  across the block. Actually wins — ~2× faster than the naive
  multi-kernel baseline at N=4096, D=64.

Benchmarked against PyTorch's `scaled_dot_product_attention` as a
ceiling (PyTorch picks FlashAttention-2 + Tensor Cores on Ada).

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
**< 2.4e-8 for every case** (v1 and v2). Within single-precision noise.

## Benchmark (RTX 4080 Laptop, Ada SM 8.9, FP32)

| Shape | Naive (3-kernel) | Flash v1 | **Flash v2 (Q-tile)** | PyTorch FP32 | PyTorch FP16 |
|---|--:|--:|--:|--:|--:|
| N=1024 D=64  | 0.42 ms | 1.17 ms | **0.28 ms** | 0.20 ms | 0.06 ms |
| N=2048 D=64  | 1.59 ms | 4.71 ms | **0.98 ms** | 0.39 ms | 0.11 ms |
| N=4096 D=64  | 6.57 ms | 16.4 ms | **3.37 ms** | 1.18 ms | 0.28 ms |
| N=1024 D=128 | 0.69 ms | 1.52 ms | **0.66 ms** | 0.26 ms | 0.06 ms |
| N=2048 D=128 | 2.81 ms | 6.10 ms | **2.08 ms** | 0.54 ms | 0.13 ms |
| N=4096 D=128 | 12.8 ms | 24.2 ms | **7.48 ms** | 2.12 ms | 0.43 ms |

Summary at `N=4096, D=64` (FP32):
- Flash v2 is **1.95× faster** than the multi-kernel naive baseline.
- Still **~3× slower than PyTorch FP32 SDPA** (uses cuBLAS + TF32 Tensor Cores).
- **~12× slower than PyTorch FP16 SDPA** (FP16 + Tensor Cores).

## Design notes — what the v1 → v2 jump changed

Nsight Compute metrics on `flash_attn_*_kernel` at `N=2048, D=64`:

| Metric | v1 (1 row / block) | **v2 (Q-tile, 4 rows / block)** |
|---|--:|--:|
| Theoretical occupancy | 16.67% | **41.67%** |
| Achieved occupancy | 16.66% | **35.65%** |
| Compute (SM) Throughput | 36.6% | **43.8%** |
| L1 / TEX Throughput | 37.4% | **82.8%** |
| L2 Throughput | 7.9% | 18.5% |
| DRAM Throughput | 1.1% | 2.6% |
| Registers / thread | 40 | 60 |

Two things are doing the work:

1. **Per-block Q tile** — 4 output rows share the same K/V tile load. Cuts
   K/V memory traffic 4× and raises L1 cache utilisation from 37% to 83%.
2. **Exact-D templates** — `flash_attn_qtile_kernel<BR, BC, D_EXACT>`
   lets `sK/sV` size match the actual head dimension instead of a
   wasteful `D_MAX=128`. Shrinks shared memory per block, lifting the
   shmem-bound occupancy cap from 2 blocks/SM to 5.

## Remaining gap to PyTorch

- **No Tensor Cores.** PyTorch FP32 uses TF32 via cuBLAS for the matmul
  hot path; PyTorch FP16 uses full FP16 Tensor Cores. Both paths get 4-8×
  arithmetic throughput over plain FP32 FMA. That's most of the gap.
- **No FP16 data path.** Halved memory bandwidth alone buys another
  big chunk even before the Tensor Core math kicks in.

Next iteration on this repo would be `wmma`-based matmul in the score
and output stages. Not done here — the scope of this project was to
get the online-softmax algorithm working and beat the naive baseline
without hardware acceleration.

## Build & run

```bat
build.bat
build\bin\attn.exe --N 2048 --D 64
```

The binary runs naive + flash v1 + flash v2 (Q-tile), checks all three
against a CPU double-precision reference, and prints per-kernel timings.

PyTorch reference numbers:
```bash
python scripts\torch_bench.py
```

## Files

```
src/
  attention.cuh   naive kernels + flash v1 + flash v2 (Q-tile)
  main.cu         test harness: CPU ref, correctness diff, timing
scripts/
  torch_bench.py  PyTorch SDPA reference timings
docs/
  profile/flash.ncu-rep   Nsight Compute: flash v1
  profile/qtile.ncu-rep   Nsight Compute: flash v2
build.bat
```

## Hardware

RTX 4080 Laptop (Ada Lovelace, SM 8.9, 58 SMs, 12 GB, ~40 MB L2, 100 KB shmem/SM).
