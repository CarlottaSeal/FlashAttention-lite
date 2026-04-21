# FlashAttention-lite

CUDA FlashAttention forward, written to land on a kernel that actually beats
a naive multi-kernel baseline. Three kernels here, progressively more serious:

- **v1 (didactic):** FP32, 1 output row per block, 1 warp. Minimum-viable
  online softmax. Useful as a correctness reference; slower than naive.
- **v2 (Q-tile):** FP32, 4 output rows per block sharing the same K/V
  tile. Already beats the naive baseline (1.5–2.6× at N=4096).
- **v3 (wmma):** FP16 inputs, FP32 accumulate, 16×16×16 Tensor Core
  fragments for both `Q·Kᵀ` and `P·V`. ~2.6× over naive, ~1.5× over
  Q-tile at N=4096, D=64.

Benchmarked against PyTorch's `scaled_dot_product_attention` (FA-2 +
Tensor Cores under the hood on Ada) as a ceiling.

## Correctness

Max absolute difference against a double-precision CPU reference, at
`N=2048, D=64`:

| Kernel | Input dtype | max \|Δ\| | Notes |
|---|---|--:|---|
| naive      | FP32 | 1.96e-8 | FP32 noise floor |
| flash v1   | FP32 | 2.14e-8 | matches naive |
| flash v2   | FP32 | 2.14e-8 | same algo, different blocking |
| flash wmma | **FP16** | 3.78e-6 | FP16 precision (~4 ULPs), as expected |

## Benchmark (RTX 4080 Laptop, Ada SM 8.9)

All kernels on the same random Q/K/V. Times are per-call average of 10 runs
after 3 warmup iterations.

| Shape | naive | flash v1 | **flash v2 (Q-tile)** | **flash v3 (wmma)** | PyTorch FP32 | PyTorch FP16 |
|---|--:|--:|--:|--:|--:|--:|
| N=1024 D=64  | 0.42 ms | 1.17 ms | **0.28 ms** | 0.38 ms | 0.20 ms | 0.06 ms |
| N=2048 D=64  | 1.59 ms | 4.70 ms | **1.04 ms** | **0.89 ms** | 0.39 ms | 0.11 ms |
| N=4096 D=64  | 5.84 ms | 16.4 ms |   3.32 ms   | **2.21 ms** | 1.18 ms | 0.28 ms |
| N=1024 D=128 | 0.86 ms | 1.52 ms |   0.70 ms   | **0.66 ms** | 0.26 ms | 0.06 ms |
| N=2048 D=128 | 2.93 ms | 6.10 ms |   2.13 ms   | **1.48 ms** | 0.54 ms | 0.13 ms |
| N=4096 D=128 | 12.8 ms | 24.2 ms |   8.37 ms   | **6.65 ms** | 2.12 ms | 0.43 ms |

Speedups at `N=4096, D=64` (the cleanest case):

- **Flash v2 is 1.76× faster than naive** (FP32 → FP32)
- **Flash v3 (wmma) is 2.64× faster than naive** and **1.50× faster than v2**
  (FP16 Tensor Core does the matmul, softmax stays FP32)
- Still ~8× slower than PyTorch FP16 SDPA — PyTorch hides latency with
  many warps per block and uses larger Q tiles

At small `N` (1024, D=64) the wmma kernel is slower than Q-tile: the
wmma version uses 1 warp per block (BR=16 matches the fragment), and the
resulting occupancy (~14.6% theoretical, ~4.6% achieved) doesn't pay off
until there's enough arithmetic per block to hide it.

## Design progression

**v1 → v2: block structure.** v1 is 1 output row per block, 1 warp. Each
block pays a full K/V tile load on every iteration for one row of work,
and the 16.67% shmem-bound occupancy prevents the SM from hiding the
memory latency with other warps. v2 groups BR=4 output rows into one
block (1 warp per row). The K/V tile loaded into shared memory is
re-used across all 4 rows' matmuls. L1 throughput jumps 37% → 83%,
achieved occupancy 17% → 36%, kernel time drops ~5×.

**v2 → v3: Tensor Cores.** The two matmuls (`Q·Kᵀ` and `P·V`) are the
arithmetic hot path. Moving them to FP16 `wmma` fragments gives the
math 4–8× higher throughput on Ada's 4th-gen Tensor Cores. The softmax
and running (m, l, O) state remain FP32 for stability. Net effect: 1.5×
over v2 at large N. Less at small N because the kernel is 1 warp per
block to match the 16×16 fragment, and that caps occupancy.

Nsight Compute metrics at `N=2048, D=64`:

| Metric | v1 | v2 (qtile) | v3 (wmma) |
|---|--:|--:|--:|
| Input dtype | FP32 | FP32 | FP16 |
| Theoretical occupancy | 16.7% | **41.7%** | 14.6% |
| Achieved occupancy    | 16.7% | **35.7%** |  4.6% |
| Compute (SM) Throughput | 36.6% | 43.8% | 11.3% |
| L1 / TEX Throughput     | 37.4% | **82.8%** | 46.5% |
| Registers / thread | 40 | 60 | 60 |
| Kernel time | 4.70 ms | 1.04 ms | **0.89 ms** |

v3's low SM throughput is misleading — Tensor Core math doesn't land in
the regular SM throughput counter. The kernel is wall-clock faster but
looks "idle" to the standard SoL view.

## Where the remaining gap to PyTorch FP16 is

- **BR=16, 1 warp per block.** Real FA-2 uses larger Q tiles (BR=64 or
  128) with 4 warps per block. That gives more work per block, enough
  to amortise the K/V load and keep Tensor Cores fully fed.
- **No async K/V load.** Production kernels overlap the next tile's
  `cp.async` load with the current tile's compute. Haven't wired this up.
- **No swizzled shmem layout.** For 16-byte-aligned fp16 fragments
  wmma wants specific strides to avoid shmem bank conflicts. Using
  plain row-major here leaves some throughput on the floor.

Each of these is a known 1.5–2× win in the FA-2 paper/code. Combined
they'd close most of the remaining 8× gap.

## Build & run

```bat
build.bat
build\bin\attn.exe --N 2048 --D 64
```

All four GPU kernels run, check against a CPU double-precision reference,
and print per-call timings.

PyTorch reference numbers:
```bash
python scripts\torch_bench.py
```

## Files

```
src/
  attention.cuh   naive + flash v1 + flash v2 (Q-tile) + flash v3 (wmma/FP16)
  main.cu         harness: CPU ref, correctness diff, timing, FP32→FP16 conv kernel
scripts/
  torch_bench.py  PyTorch SDPA reference (FP32 + FP16)
docs/profile/
  flash.ncu-rep   Nsight Compute — flash v1
  qtile.ncu-rep   Nsight Compute — flash v2
  wmma.ncu-rep    Nsight Compute — flash v3
```

## Hardware

RTX 4080 Laptop (Ada Lovelace, SM 8.9, 58 SMs, 12 GB, ~40 MB L2, 100 KB
shmem/SM, 4th-gen Tensor Cores with FP16 / BF16 / FP8).
