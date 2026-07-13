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

The baseline these beat is `naive`: the textbook three passes, a score kernel that
writes the full N×N score matrix to HBM, then a softmax kernel, then an output
kernel. That N×N round-trip to global memory is exactly what the FlashAttention
variants avoid.

PyTorch's `scaled_dot_product_attention` (FA-2 + Tensor Cores under the hood)
is the ceiling I benchmark against. A Triton port (`triton/flash_triton.py`)
re-expresses the same forward; see the Triton port section below.

## Where this fits in a Transformer

Attention is the core operator of every Transformer layer: a layer is multi-head
attention followed by a feed-forward network, with residual connections and a norm.
This kernel implements that operator's forward pass, `softmax(Q*K^T / sqrt(d)) * V`,
the part that dominates memory traffic at long sequence length. That is the whole
reason FlashAttention exists.

> Scope, stated honestly: this is the encoder-style attention forward (non-causal),
> single-head on the CUDA side and multi-head in the Triton port. Add a causal mask,
> the feed-forward network, the norms, and the backward pass, and it becomes a full
> GPT block.

## Online softmax

Softmax of a row is `softmax(x)_j = exp(x_j) / sum_k exp(x_k)`. In practice you
subtract the row max `m` first, `exp(x_j - m) / sum_k exp(x_k - m)`, which is the
same value (the `exp(-m)` cancels top and bottom) but stops `exp` from overflowing
to inf on large scores.

FlashAttention streams over K/V in tiles, so it never holds a whole row at once and
cannot know the true row max up front. It keeps three running quantities per Q row:
the max seen so far `m`, the running denominator `l`, and the output accumulator `O`.

The name is the algorithms sense of *online*: consume the stream incrementally with
no second pass, nothing web-related. The technique predates FlashAttention (Milakov
and Gimelshein, 2018); this repo implements it, it is not my invention.

When a new tile produces a larger max `m_new > m_old`, everything summed so far used
a stale max and has to be corrected. The fix-up factor falls out of one identity:

```
exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new)
                                  \_________________/
                                        alpha
```

So before folding in the new tile, rescale the old state by `alpha`:

```
m_new = max(m_old, rowmax(tile))
alpha = exp(m_old - m_new)
l     = alpha * l + sum(exp(tile - m_new))
O     = alpha * O + P_tile @ V_tile
```

After the last tile, divide once: `O = O / l`. No N-wide score row ever exists, no
second pass over the scores, and the result is numerically equivalent to the naive
two-pass softmax.

Worked example, a row of scores `[1, 3]` arriving as two tiles, `1` then `3`:

```
tile 1 (sees 1):  m = 1,  l = exp(1-1) = 1,        O = 1 * V1
tile 2 (sees 3):  m_new = max(1, 3) = 3
                  alpha = exp(1 - 3) = 0.135
                  l = 0.135 * 1 + exp(3-3) = 1.135
                  O = 0.135 * O + 1 * V2
final:            O = O / l = (0.135 * V1 + V2) / 1.135
```

A one-shot softmax of `[1, 3]` gives weights `[exp(-2), 1] / 1.135 = [0.119, 0.881]`
and the same `O`. The streaming pass with the `alpha` rescale is exact, not an
approximation.

In the code: `attention.cuh:169-171` (m_new, alpha), `:197` (l update), `:204` (the
`alpha * O` rescale).

## Tensor Cores and wmma

v3 and v4 run the two matmuls (Q*K^T and P*V) on Tensor Cores through the `wmma`
API (`#include <mma.h>`). A `wmma::fragment` is a small matrix tile spread across the
32 lanes of one warp; the warp issues a single `mma_sync` that multiplies two input
fragments and accumulates into a third. Every fragment here is the `16x16x16` shape:
a 16x16 output tile built from a 16-wide contraction.

The fragment types pin the precision. The two inputs (`matrix_a`, `matrix_b`) are
`__half` (FP16); the accumulator is `float` (FP32). That is the "FP16 in, FP32
accumulate" the design notes mention, and it is a hardware feature, not something the
kernel emulates: the multiply runs at FP16 Tensor Core rate while the running sum
keeps FP32 range, so accumulating hundreds of products does not lose precision.
Softmax (`exp`, the row max, the denominator) stays FP32 for the same reason.

Why the accumulator specifically: summing many products is where low precision hurts,
because a large running total swallows small new terms. FP16 near 1000 has a step of
about 0.5, so `1000 + 0.01` rounds straight back to `1000` and the small term is lost;
FP32 keeps it. The inputs can be FP16, but what they accumulate into sets the precision
floor, so it stays FP32.

Two details worth knowing. For Q*K^T the K fragment is loaded `col_major`, which
transposes it for free, so no explicit K^T buffer is needed. And a `D=64` head is
contracted as four 16-wide tiles (`D / 16`), looping `mma_sync` four times into the
same accumulator. Code: `attention.cuh:399-409` (Q*K^T) and `:439-450` (P*V).

## How it's measured

Correctness and speed are checked separately, by different tools. A kernel can be
correct and slow (v1 is both).

- **Correctness.** `main.cu` computes a reference on the CPU in double precision
  (`attention_cpu_ref`), then takes the max absolute elementwise difference against
  it for each kernel (`diff_stats`). The Correctness table below is that max |Δ|.
- **Speed.** `time_kernel_ms` runs 3 warmup launches, then times 10 back-to-back
  launches with one `cudaEvent` pair and divides by 10. The "Nx over naive" figures
  are just `t_naive / t_kernel`. A fixed RNG seed keeps every run reproducible.

The Nsight metrics further down come from a separate `ncu` profiling run, not from
this timing harness.

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

Each is a known 1.5-2x win. All three are memory-access and scheduling
optimizations: the online-softmax math and the tiling already match what production
FlashAttention does, so the remaining gap is implementation, not the algorithm.

## Triton port

`triton/flash_triton.py` re-expresses the same forward in Triton: the identical
online-softmax tiling, but Triton owns the warp and shared-memory layout and lowers
the two matmuls to Tensor Cores through `tl.dot`. Multi-head, with Q/K/V laid out
`[Z, H, N, D]`, non-causal. What the CUDA kernels do by hand (fragment loads, shmem
staging, the dynamic-shmem opt-in), `tl.dot` and Triton's scheduler do
automatically: same algorithm, different front-end.

Validated against `torch.scaled_dot_product_attention` to max |Δ| ~6e-5 (FP16).

```bash
python triton/flash_triton.py
```

On this Windows torch build the flash / cuDNN SDPA backends aren't compiled in, so
SDPA falls back to the mem-efficient kernel. The Triton timing beats that, not
production FlashAttention; benchmarking against true flash needs Linux + flash-attn.

## Requirements

- An NVIDIA GPU. The `wmma` kernels need Tensor Cores (SM 7.0+); developed on Ada
  (SM 8.9) and Blackwell (SM 12.0). The FP32 paths (naive, v1, v2) run anywhere.
- CUDA Toolkit: 13.2 for the Blackwell build, 12.x for the Ada build.
- References only: a CUDA `torch` build, plus `triton` (`triton-windows` on Windows)
  for the Triton port.

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
triton/
  flash_triton.py Triton port, multi-head, validated vs torch SDPA
docs/profile/
  flash.ncu-rep / qtile.ncu-rep / wmma.ncu-rep / wmma_v4.ncu-rep   Nsight Compute, v1-v4
```

## Scope and limitations

A deliberately narrow study kernel, kept small so each optimization is isolated and
measurable:

- **Forward pass only.** No backward; that needs activation recomputation and
  dQ/dK/dV and is meaningfully harder.
- **Single head, non-causal** on the CUDA side. The Triton port is multi-head
  (`[Z, H, N, D]`) but also non-causal, so no causal mask and no skipping of
  fully-masked tiles.
- **`D` in {64, 128}, a multiple of 16** to match the wmma tile width. Other head
  dimensions are unsupported on the wmma paths.
- **FP16 inputs on v3/v4**, FP32 accumulate. No FP8 path yet.
- **Single GPU.** No multi-GPU or distributed.

The goal was to take one kernel from a correct baseline to ~4x by profiling one
bottleneck at a time, not to reimplement production FlashAttention-2.

## Hardware

Developed on RTX 4080 Laptop (Ada, SM 8.9, 58 SMs, 12 GB, ~40 MB L2, 100 KB
shmem/SM). Re-benchmarked on RTX 5070 Ti (Blackwell, SM 12.0).
