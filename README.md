# FlashAttention-lite

CUDA FlashAttention forward, written to land on a kernel that actually beats
a naive multi-kernel baseline. Four kernels here, progressively more serious:

- **v1 (didactic):** FP32, 1 output row per block, 1 warp. Minimum-viable
  online softmax. Useful as a correctness reference; slower than naive.
- **v2 (Q-tile):** FP32, 4 output rows per block sharing the same K/V
  tile. Already beats the naive baseline (1.5–2.6× at N=4096).
- **v3 (wmma):** FP16 inputs, FP32 accumulate, 16×16×16 Tensor Core
  fragments for both `Q·Kᵀ` and `P·V`. ~2.6× over naive, ~1.5× over
  Q-tile at N=4096, D=64. One warp per block, BR=16.
- **v4 (wmma, BR=64):** Same Tensor Core matmuls, but 4 warps per block
  cooperating on the K/V load and each owning a 16-row Q slice. K/V tile
  is loaded once per block and reused 4× across the warps. Uses dynamic
  shared memory (opt-in via `cudaFuncSetAttribute`) because at D=128,
  BR=64 needs 63 KB of shmem — above ptxas's 48 KB static ceiling.
  **4.11× over naive, 1.59× over v3** at N=4096, D=64.

Benchmarked against PyTorch's `scaled_dot_product_attention` (FA-2 +
Tensor Cores under the hood on Ada) as a ceiling.

## Correctness

Max absolute difference against a double-precision CPU reference, at
`N=2048, D=64`:

| Kernel | Input dtype | max \|Δ\| | Notes |
|---|---|--:|---|
| naive         | FP32 | 1.96e-8 | FP32 noise floor |
| flash v1      | FP32 | 2.14e-8 | matches naive |
| flash v2      | FP32 | 2.14e-8 | same algo, different blocking |
| flash v3 wmma | **FP16** | 3.78e-6 | FP16 precision (~4 ULPs), as expected |
| flash v4 wmma | **FP16** | 3.78e-6 | matches v3 bit-for-bit |

## Benchmark (RTX 4080 Laptop, Ada SM 8.9)

All kernels on the same random Q/K/V. Times are per-call average of 10 runs
after 3 warmup iterations.

| Shape | naive | flash v1 | flash v2 (Q-tile) | flash v3 (wmma BR=16) | **flash v4 (wmma BR=64)** | PyTorch FP32 | PyTorch FP16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| N=1024 D=64  |  0.47 ms |  1.38 ms | 0.30 ms | 0.42 ms | **0.29 ms** | 0.20 ms | 0.06 ms |
| N=2048 D=64  |  1.57 ms |  4.45 ms | 1.01 ms | 0.85 ms | **0.50 ms** | 0.39 ms | 0.11 ms |
| N=4096 D=64  |  6.37 ms | 17.5 ms  | 3.71 ms | 2.47 ms | **1.55 ms** | 1.18 ms | 0.28 ms |
| N=1024 D=128 |  1.10 ms |  2.48 ms | 1.00 ms | 1.00 ms | **0.65 ms** | 0.26 ms | 0.06 ms |
| N=2048 D=128 |  3.27 ms |  6.78 ms | 2.28 ms | 1.59 ms | **0.86 ms** | 0.54 ms | 0.13 ms |
| N=4096 D=128 | 12.96 ms | 26.2 ms  | 8.18 ms | 7.25 ms | **3.37 ms** | 2.12 ms | 0.43 ms |

Speedups at `N=4096, D=64`:

- **Flash v2 is 1.72× faster than naive** (FP32 → FP32)
- **Flash v3 (wmma BR=16) is 2.58× faster than naive** and **1.50× faster
  than v2** (FP16 Tensor Core on the matmul, softmax stays FP32)
- **Flash v4 (wmma BR=64) is 4.11× faster than naive** and **1.59× faster
  than v3** — same Tensor Core math, but 4 warps per block amortise the
  K/V load and raise achieved occupancy from ~4.6% (v3) to ~8.3% (v4)
- Still ~5.5× slower than PyTorch FP16 SDPA (v4 vs PyTorch FP16) — the
  remaining gap is `cp.async` overlap of K/V load with compute and
  swizzled shmem layouts for conflict-free wmma loads

At small `N=1024` v3 can be slower than v2 (1 warp / block under-uses the
SM); v4 wins in every shape because the larger block does 4× the work per
K/V load.

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

**v3 → v4: BR=64 with 4 warps per block.** v3's BR=16, 1-warp-per-block
leaves money on the table twice over: only 4.6% achieved occupancy, and
the same K/V tile gets loaded separately by every block for 16 Q rows of
work. v4 packs 4 warps into one block. All four cooperate on the K/V
load (4× fewer loads per row of work), then each warp runs its own
wmma `Q_slice · Kᵀ` and `P_slice · V` on its 16-row Q slice. L1/TEX
throughput rises 46% → 64%, achieved occupancy 4.6% → 8.3%. Net
1.5–2.2× over v3 depending on shape.

At D=128, BR=64 needs 63 KB of shared memory per block, above ptxas's
48 KB static ceiling. v4 uses `extern __shared__` with
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
at launch to opt in to Ada's 100 KB/SM cap.

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

v3 and v4's low SM throughput are misleading — Tensor Core math doesn't
land in the regular SM throughput counter. The kernels are wall-clock
faster but look "idle" to the standard SoL view.

v4 is still shmem-occupancy-limited (block limit from shmem = 2/SM); the
next step that could raise occupancy further would be cutting the per-block
shmem via `cp.async` pipelining (overlap tile N+1's load with tile N's
compute instead of double-buffering both in shmem).

## Where the remaining gap to PyTorch FP16 is

At N=4096, D=64 v4 is 1.55 ms vs PyTorch's 0.28 ms — a ~5.5× gap.
Known wins from the FA-2 paper/code:

- **`cp.async` K/V load.** Production kernels overlap the next tile's
  async copy with the current tile's compute. Also frees up one shmem
  buffer, which is the current occupancy limiter. Not wired up.
- **Swizzled shmem layout.** For 16-byte-aligned FP16 fragments `wmma`
  wants specific strides to avoid shmem bank conflicts; plain row-major
  here leaves L1 throughput on the floor.
- **Split-K / 8 warps per block.** At very large N, splitting the Q
  tile across more warps and reducing after unlocks more parallelism.

Each is a known 1.5–2× win.

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
  attention.cuh   naive + flash v1 + v2 (Q-tile) + v3 (wmma BR=16) + v4 (wmma BR=64)
  main.cu         harness: CPU ref, correctness diff, timing, FP32→FP16 conv kernel
scripts/
  torch_bench.py  PyTorch SDPA reference (FP32 + FP16)
docs/profile/
  flash.ncu-rep   Nsight Compute — flash v1
  qtile.ncu-rep   Nsight Compute — flash v2
  wmma.ncu-rep    Nsight Compute — flash v3
  wmma_v4.ncu-rep Nsight Compute — flash v4
```

## Hardware

RTX 4080 Laptop (Ada Lovelace, SM 8.9, 58 SMs, 12 GB, ~40 MB L2, 100 KB
shmem/SM, 4th-gen Tensor Cores with FP16 / BF16 / FP8).
