# FlashAttention-lite

CUDA FlashAttention forward pass. Four kernels, each one fixing the bottleneck
the profiler found in the last:

- **v1 (didactic):** FP32, 1 output row per block, 4 warps (128 threads). Basic online
  softmax. Correctness reference, slower than naive.
- **v2 (Q-tile):** FP32, 4 output rows per block sharing one K/V tile. Beats
  naive (1.5-2.6x at N=4096).
- **v3 (wmma):** FP16 inputs, FP32 accumulate, 16x16x16 Tensor Core fragments
  for both Q*K^T and P*V. ~2.6x over naive, ~1.5x over Q-tile at N=4096, D=64.
  One warp per block, BR=16.
- **v4 (wmma, BR=64):** Same Tensor Core matmuls, 4 warps per block. They
  cooperate on the K/V load and each owns a 16-row Q slice, so the K/V tile is
  loaded once and reused 4x. Needs dynamic shared memory (opt-in via
  `cudaFuncSetAttribute`): at D=128, BR=64 wants 63 KB, above ptxas's 48 KB
  static ceiling. **4.11x over naive, 1.59x over v3** at N=4096, D=64.

PyTorch's `scaled_dot_product_attention` (FA-2 + Tensor Cores under the hood)
is the ceiling I benchmark against.

## Correctness

Max absolute difference against a double-precision CPU reference, at
`N=2048, D=64`:

| Kernel | Input dtype | max \|Δ\| | Notes |
|---|---|--:|---|
| naive         | FP32 | 1.96e-8 | FP32 noise floor |
| flash v1      | FP32 | 2.14e-8 | matches naive |
| flash v2      | FP32 | 2.14e-8 | same algo, different blocking |
| flash v3 wmma | **FP16** | 3.78e-6 | FP16 precision (~4 ULPs) |
| flash v4 wmma | **FP16** | 3.78e-6 | matches v3 bit-for-bit |

## Benchmark (RTX 5070 Ti, Blackwell SM 12.0)

Rebuilt with `-arch=sm_120` on CUDA 13.2. Same harness, same random Q/K/V,
per-call average of 10 runs after 3 warmups. PyTorch is `torch 2.11.0+cu128`.

| Shape | naive | flash v1 | flash v2 (Q-tile) | flash v3 (wmma BR=16) | **flash v4 (wmma BR=64)** | PyTorch FP32 | PyTorch FP16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| N=2048 D=64  | 1.003 ms |  4.806 ms | 0.659 ms | 0.620 ms | **0.457 ms** | 0.211 ms | 0.093 ms |
| N=4096 D=64  | 4.099 ms | 18.521 ms | 2.407 ms | 1.678 ms | **0.967 ms** | 0.428 ms | 0.181 ms |
| N=4096 D=128 | 7.837 ms | 29.393 ms | 5.677 ms | 3.645 ms | **1.490 ms** | 0.691 ms | 0.263 ms |

At N=4096, D=64: v2 is 1.70x over naive, v3 2.44x, v4 4.24x over naive and
1.73x over v3. At D=128 v4 reaches 5.26x over naive. v4 is still ~5.3x behind
PyTorch FP16 SDPA (0.967 vs 0.181 ms), same gap shape as on Ada; the remaining
wins are `cp.async` load/compute overlap, swizzled shmem, and split-K.
Correctness holds on Blackwell (FP32 ~2e-8, FP16 wmma ~2e-6).

## Benchmark (RTX 4080 Laptop, Ada SM 8.9)

Same harness, per-call average of 10 runs after 3 warmups.

| Shape | naive | flash v1 | flash v2 (Q-tile) | flash v3 (wmma BR=16) | **flash v4 (wmma BR=64)** | PyTorch FP32 | PyTorch FP16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| N=1024 D=64  |  0.47 ms |  1.38 ms | 0.30 ms | 0.42 ms | **0.29 ms** | 0.20 ms | 0.06 ms |
| N=2048 D=64  |  1.57 ms |  4.45 ms | 1.01 ms | 0.85 ms | **0.50 ms** | 0.39 ms | 0.11 ms |
| N=4096 D=64  |  6.37 ms | 17.5 ms  | 3.71 ms | 2.47 ms | **1.55 ms** | 1.18 ms | 0.28 ms |
| N=1024 D=128 |  1.10 ms |  2.48 ms | 1.00 ms | 1.00 ms | **0.65 ms** | 0.26 ms | 0.06 ms |
| N=2048 D=128 |  3.27 ms |  6.78 ms | 2.28 ms | 1.59 ms | **0.86 ms** | 0.54 ms | 0.13 ms |
| N=4096 D=128 | 12.96 ms | 26.2 ms  | 8.18 ms | 7.25 ms | **3.37 ms** | 2.12 ms | 0.43 ms |

At N=4096, D=64: v2 1.72x over naive (FP32), v3 2.58x over naive and 1.50x over
v2 (FP16 Tensor Core on the matmul, softmax stays FP32), v4 4.11x over naive and
1.59x over v3 (4 warps amortise the K/V load, achieved occupancy 4.6% -> 8.3%).
v4 is ~5.5x behind PyTorch FP16. At small N=1024 v3 can lose to v2 (1 warp per
block under-uses the SM); v4 wins every shape because the larger block does 4x
the work per K/V load.

## Design progression

**v1 -> v2: block structure.** v1 is 1 output row per block, 4 warps (128 threads). Each block
pays a full K/V tile load every iteration for one row of work, and 16.67%
shmem-bound occupancy stops the SM from hiding the memory latency behind other
warps. v2 groups BR=4 rows into one block (1 warp per row) and reuses the
shared-memory K/V tile across all 4 rows' matmuls. L1 throughput 37% -> 83%,
achieved occupancy 17% -> 36%, kernel time ~5x faster.

**v2 -> v3: Tensor Cores.** The two matmuls (Q*K^T and P*V) are the arithmetic
hot path. Moving them to FP16 `wmma` fragments gives the math 4-8x higher
throughput on Ada's 4th-gen Tensor Cores. Softmax and the running (m, l, O)
state stay FP32 for stability. Net 1.5x over v2 at large N, less at small N
because 1 warp per block (to match the 16x16 fragment) caps occupancy.

**v3 -> v4: BR=64, 4 warps per block.** v3's BR=16 1-warp-per-block wastes work
two ways: only 4.6% achieved occupancy, and the same K/V tile is loaded
separately by every block for 16 Q rows. v4 packs 4 warps into one block, all
four cooperate on the K/V load (4x fewer loads per row), then each warp runs its
own wmma Q_slice*K^T and P_slice*V on its 16-row slice. L1/TEX 46% -> 64%,
achieved occupancy 4.6% -> 8.3%. Net 1.5-2.2x over v3 depending on shape.

At D=128, BR=64 needs 63 KB of shared memory per block, above ptxas's 48 KB
static ceiling. v4 uses `extern __shared__` plus
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
at launch to opt in to the 100 KB/SM cap.

Nsight Compute metrics at `N=2048, D=64`:

| Metric | v1 | v2 (qtile) | v3 (wmma BR=16) | **v4 (wmma BR=64)** |
|---|--:|--:|--:|--:|
| Input dtype | FP32 | FP32 | FP16 | FP16 |
| Block size | 128 | 128 | 32 | **128** |
| Theoretical occupancy | 16.7% | **41.7%** | 14.6% | 16.7% |
| Achieved occupancy    | 16.7% | **35.7%** |  4.6% |  **8.3%** |
| Compute (SM) Throughput | 36.6% | 43.8% | 11.3% |  7.5% |
| L1 / TEX Throughput     | 37.4% | **82.8%** | 46.5% | 63.9% |
| Registers / thread | 40 | 60 | 60 | 57 |
| Shared memory / block | ~20 KB | ~18 KB | ~14 KB | **35 KB dyn** |
| Kernel time | 4.70 ms | 1.04 ms | 0.89 ms | **0.56 ms** |

Watch out reading v3/v4: the low SM throughput is misleading. Tensor Core math
doesn't land in the regular SM throughput counter, so the kernels look idle in
the standard SoL view even though they finish faster. v4 is still
shmem-occupancy-limited (block limit from shmem = 2/SM); the next occupancy win
is cutting per-block shmem via `cp.async` pipelining (overlap tile N+1's load
with tile N's compute instead of double-buffering both).

## Remaining gap to PyTorch FP16

At N=4096, D=64 v4 is 1.55 ms vs PyTorch's 0.28 ms on Ada, ~5.5x. Known wins
from the FA-2 paper and code:

- **`cp.async` K/V load.** Production kernels overlap the next tile's async copy
  with the current tile's compute. Also frees one shmem buffer, which is the
  current occupancy limiter. Not wired up.
- **Swizzled shmem layout.** 16-byte-aligned FP16 `wmma` fragments want specific
  strides to avoid bank conflicts; plain row-major leaves L1 throughput on the
  floor.
- **Split-K / 8 warps per block.** At very large N, splitting the Q tile across
  more warps and reducing after unlocks more parallelism.

Each is a known 1.5-2x win.

## Build & run

```bat
build_5070.bat                  REM Blackwell (sm_120, CUDA 13.2). build.bat is the older Ada/sm_89 setup.
build\bin\attn.exe --N 2048 --D 64
```

All four GPU kernels run, check against a CPU double-precision reference, and
print per-call timings.

PyTorch reference numbers:
```bash
python scripts\torch_bench.py
```

## Files

```
src/
  attention.cuh   naive + flash v1 + v2 (Q-tile) + v3 (wmma BR=16) + v4 (wmma BR=64)
  main.cu         harness: CPU ref, correctness diff, timing, FP32->FP16 conv kernel
scripts/
  torch_bench.py  PyTorch SDPA reference (FP32 + FP16)
docs/profile/
  flash.ncu-rep / qtile.ncu-rep / wmma.ncu-rep / wmma_v4.ncu-rep   Nsight Compute, v1-v4
```

## Hardware

Developed on RTX 4080 Laptop (Ada, SM 8.9, 58 SMs, 12 GB, ~40 MB L2, 100 KB
shmem/SM). Re-benchmarked on RTX 5070 Ti (Blackwell, SM 12.0).
