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

> Scope, stated honestly: the CUDA side is the attention forward, single-head and
> non-causal. The Triton port is multi-head, optionally causal, with a backward
> pass. Add the feed-forward network and the norms, and it becomes a full GPT block.

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
`[Z, H, N, D]`, causal or not. What the CUDA kernels do by hand (fragment loads, shmem
staging, the dynamic-shmem opt-in), `tl.dot` and Triton's scheduler do
automatically: same algorithm, different front-end.

Validated against `torch.scaled_dot_product_attention` to max |Δ| ~1e-4 (FP16),
~5e-4 causal. Causal is looser because early rows average only a few V rows, so
the outputs are larger and so is an FP16 ULP.

```bash
python triton/flash_triton.py
```

On this Windows torch build the flash / cuDNN SDPA backends aren't compiled in, so
SDPA falls back to the mem-efficient kernel. The Triton timing beats that, not
production FlashAttention; benchmarking against true flash needs Linux + flash-attn.

### Causal mask and tile skip

`flash_attn(q, k, v, causal=True)` masks `S[i, j]` to `-inf` where `j > i`. Done
naively, every K/V tile still gets loaded, multiplied, and then mostly thrown
away. For a Q tile covering rows `[m0, m0 + BLOCK_M)`, K/V tiles fall into three
groups:

- columns entirely `< m0`: fully visible, no mask needed
- the diagonal tile(s), `[m0, m0 + BLOCK_M)`: masked elementwise
- columns entirely `>= m0 + BLOCK_M`: fully masked, so never loaded

The kernel runs the inner loop twice, `[0, m0)` with `MASK=False` and the
diagonal with `MASK=True`, and stops there. `MASK` is a `constexpr`, so the
unmasked pass compiles without the `tl.where`. `BLOCK_M % BLOCK_N == 0` keeps
the diagonal on a tile boundary. `skip=False` runs the masked loop over every
tile, kept for comparison.

FP16, `Z=1, H=16`, RTX 5070 Ti, ms:

| N, D | non-causal | causal, no skip | causal, skip | skip / non-causal | SDPA causal |
|---|--:|--:|--:|--:|--:|
| 1024, 64 | 0.059 | 0.060 | 0.048 | 0.81 | 0.095 |
| 2048, 64 | 0.240 | 0.245 | 0.135 | 0.56 | 0.289 |
| 4096, 64 | 0.817 | 0.843 | 0.468 | 0.57 | 0.957 |
| 8192, 64 | 3.311 | 3.387 | 1.758 | 0.53 | 3.454 |
| 4096, 128 | 2.158 | 2.369 | 1.277 | 0.59 | 2.051 |

Masking without skipping costs 2-10% over non-causal: the extra `tl.where` and
nothing saved. With skipping, the tiles visited are `T(T+1)/2` out of `T^2` for
`T = N / 64` Q tiles, a ratio of 0.504 at N=8192, and the measured ratio is 0.53.
At N=1024 the grid is only 256 CTAs and the ratio drops to 0.81, since launch
overhead and the tail are a bigger share of a 50 us kernel.

Q tiles are uneven: the last one walks all `T` tiles and the first walks one.
I tried launching them in reverse, heaviest first, to shorten the tail. The
difference was within noise (±5%). With `H=16` the grid's fastest dimension
already interleaves light and heavy tiles across heads in every wave.

### Autotuning

`flash_attn_tuned` wraps the same kernel in `@triton.autotune` over 24 configs,
`BLOCK_M` {64, 128} x `BLOCK_N` {32, 64} x `num_warps` {4, 8} x `num_stages`
{2, 3, 4}, keyed on `(N, D, CAUSAL)`. `triton/autotune_sweep.py` times every
config on every shape (`Z=1, H=16`, N 1024-8192, D 64/128, both modes) with
`do_bench` and logs registers, spills and shared memory to
`docs/autotune_sweep.csv`.

Winners, with how far the hand-picked default (64/64, 4 warps, 3 stages) lands
behind:

| D | mode | winner (BM/BN/warps/stages) | default vs winner |
|---|---|---|--:|
| 64 | non-causal | flat: top 3 within 1.4% at every N | 1.01-1.03x |
| 64 | causal | 64/64/4/3 at every N | 1.00x |
| 128 | non-causal | 128/64/8/2-3 (128/32/4/2 at N=8192) | 1.22-1.30x |
| 128 | causal | `BLOCK_M=128` at every N, mostly 128/64/8/3 | 1.05-1.33x |

**The axis that matters is D, not causal.** At D=64 almost any sane config is
within a few percent, and the causal winner is the default. At D=128 the winner
moves to `BLOCK_M=128` in both modes. Rows per warp stay at 16 either way
(128 / 8 warps), but every K/V tile loaded into shmem now serves 128 Q rows
instead of 64, and at D=128 a K/V tile is twice the bytes, so that reuse is
worth 20-30%. Causal vs non-causal shifts the nominal winner between configs
that are within noise of each other.

**The worst configs are 2x slower, and it's a layout problem.** `BLOCK_M=64`
with 8 warps is the bottom of the table for both D. The TTGIR shows
`warpsPerCTA = [8, 1]` with `instrShape = [16, 8]`: eight warps along M need
128 rows, but the tile has 64, so warps 4-7 compute a copy of what warps 0-3
compute. Per-warp SASS is the same as the 4-warp version (64 HMMA), there are
just twice as many warps doing it.

**`num_stages`.** Mean time against stages=2, over all 64 runs of each:

| | stages=3 | stages=4 |
|---|--:|--:|
| D=64 | 0.95x | 1.00x |
| D=128 | 1.04x | 1.13x (half the configs don't fit) |

Going from no pipelining to 2 stages is the big step (A1: 0.044 -> 0.040 ms). A
third buffer helps at D=64 and hurts at D=128. Each stage at `BLOCK_N=64, D=128`
is 32 KB of K+V. Every `BLOCK_N=64` config with 4 stages goes over the 99 KB
per-block limit and fails to compile, and `128/64/s3` sits at 96 KB. Occupancy doesn't explain the s2/s3
difference here, since both are 1 CTA/SM by registers (220 x 256 threads). I
didn't profile further.

**Spills aren't automatically fatal.** `128/32` with 4 warps at D=128 is 255
registers and spills 10 (non-causal) or 30 (causal), yet it wins at N=8192 in
both modes. I haven't profiled why. The spill count is small and spills land in
L1 first.

**The tuner's own picks.** `@triton.autotune` agreed with the sweep's winner,
or picked something within 2.6% of it, on 15 of 16 shapes. The exception was
causal N=1024 D=128 (a 0.11 ms kernel), 9% off. On the sub-0.2 ms shapes,
run-to-run noise is the same size as the gap between the top configs.

```bash
python triton/autotune_sweep.py
```

### Backward

`attention(q, k, v, causal, bwd='flash' | 'naive')` is a `torch.autograd.Function`.
The forward saves `lse = m + log(l)` per row (one float per row, `[Z, H, N]`), so
the backward can rebuild any tile of `P` in one step as `exp(S - lse)` without
redoing the online max/sum.

The math, per head, with `S = scale * Q K^T` and `P = softmax(S)`:

```
dV = P^T dO
dP = dO V^T
dS = P * (dP - rowsum(P * dP))
dQ = scale * dS K
dK = scale * dS^T Q
```

**Naive** (`flash_bwd.py: flash_bwd_naive`) does exactly that, one kernel per
line: a batched matmul kernel for the five products, a row kernel for `P` and one
for `dS`, with `S/P` and `dP/dS` materialized as `[B, N, N]` fp32. Matmuls use
`input_precision="ieee"` so it stays an exact reference.

**Tiled** (`flash_bwd`) never builds an N x N matrix. Three kernels:

1. `_delta`: `Delta_i = rowsum(dO_i * O_i)`. This equals `rowsum(P_i * dP_i)`,
   since `sum_j P_ij (dO_i . V_j) = dO_i . sum_j P_ij V_j = dO_i . O_i`, but costs
   O(N D) instead of O(N^2).
2. `_bwd_dkdv`: one program per K/V tile. K and V stay in registers. It walks the Q
   tiles, recomputes `P^T` for the tile from `K Q^T` and `lse`, and accumulates `dV
   += P^T dO` and `dK += dS^T Q`.
3. `_bwd_dq`: one program per Q tile, walks the K/V tiles and accumulates `dQ +=
   dS K`.

dQ needs a sum over K tiles and dK/dV need a sum over Q tiles, so one loop order
can't own all three. FlashAttention-2 parallelizes over K/V tiles and
accumulates dQ with atomic adds. I split
dQ into its own kernel instead, which keeps it deterministic and atomic-free, and
pays for it by recomputing `Q K^T` and `dO V^T` a second time: 7 matmuls instead
of 5. Causal skips tiles the same way as the forward. The dK/dV kernel starts at
the diagonal Q tile, the dQ kernel stops at it, and only the diagonal tile is
masked.

Correctness: `torch.autograd.gradcheck` passes in fp64 for both versions,
causal and not, at N=40 and N=100 (tail tiles). The forward accumulates in fp64
when given fp64 inputs so finite differences hold. In fp16, against an fp32
PyTorch reference at N=1024, D=64:

| | dq | dk | dv |
|---|--:|--:|--:|
| tiled, non-causal | 2.2e-4 | 2.6e-4 | 2.3e-4 |
| SDPA backward, non-causal | 2.4e-4 | 2.5e-4 | 2.3e-4 |
| tiled, causal | 1.4e-3 | 1.9e-3 | 1.9e-3 |
| SDPA backward, causal | 1.4e-3 | 2.0e-3 | 1.9e-3 |

The tiled error is FP16 rounding of `P` and `dS` before they go into `tl.dot`,
the same as PyTorch's own fp16 backward. Causal is larger because early keys
collect gradient from many rows, so the values are bigger.

Backward only, `Z=1, H=16`, FP16, ms (peak extra memory in MB):

| N | D | causal | naive | tiled | SDPA |
|--:|--:|:-:|--:|--:|--:|
| 1024 | 64 | | 1.72 (146) | **0.21** (6) | 0.40 (12) |
| 4096 | 64 | | 27.4 (2120) | **2.73** (24) | 5.44 (48) |
| 8192 | 64 | | - | **10.8** (48) | 22.0 (97) |
| 4096 | 128 | | 45.9 (2192) | **5.93** (48) | 13.0 (108) |
| 8192 | 128 | | - | **23.4** (96) | 51.2 (205) |
| 4096 | 64 | yes | 27.5 (2120) | **1.52** (24) | 3.09 (48) |
| 8192 | 64 | yes | - | **5.60** (48) | 11.6 (97) |
| 8192 | 128 | yes | - | **11.9** (96) | 26.0 (205) |

Full grid in `python triton/bwd_bench.py`. Things the numbers show:

- **Memory is the reason tiling exists.** Naive needs two `[B, N, N]` fp32
  buffers. That's 2 GB at N=4096 and would be 8 GB at N=8192, so it's skipped
  there. Tiled needs `Delta` plus the outputs.
- **The 10x over naive isn't all tiling.** Naive also runs fp32 IEEE matmuls
  without Tensor Cores, and causal doesn't help it at all, since it computes the
  full matrix and zeroes half.
- **Backward / forward = 3.3x** (2.73 vs 0.83 ms at N=4096, D=64), close to the 7
  matmuls vs 2 ratio. Causal tiled lands at 0.51-0.56x of non-causal from N=4096
  up (0.69-0.73x at N=1024), the same pattern as the forward.
- **2x faster than SDPA here, with the usual caveat.** On this Windows build SDPA's
  backward is the mem-efficient kernel, not FlashAttention-2.

At D=128 the backward runs 2 pipeline stages instead of 3. Three stages want
105 KB of shmem, over the 99 KB limit.

```bash
python triton/test_bwd.py    # gradcheck + fp16 error
python triton/bwd_bench.py
```

### Precision: FP8 and BF16 vs FP16

sm_120 takes FP8 in `tl.dot`. The same forward kernel, fed `float8_e4m3fn` inputs,
lowers to `QMMA.16832.F32.E4M3` (k=32 per instruction, twice FP16's `HMMA.16816`).
P is cast to the input dtype before `P @ V`, so it's FP8 too. Accumulation stays
FP32 and the output is FP16. No scaling factors: `randn` inputs sit well inside
e4m3's range (max 448).

Forward only, Z=1, H=16, best of three tile configs per dtype. Error is mean |d|
against fp32 SDPA on the original fp32 inputs, so it includes input rounding.

| N | D | causal | FP16 ms | BF16 | e4m3 | e5m2 | FP16 err | BF16 err | e4m3 err | e5m2 err |
|--:|--:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 64 | | 0.063 | 0.96x | 1.29x | 1.28x | 1.9e-5 | 1.5e-4 | 2.2e-3 | 4.3e-3 |
| 8192 | 64 | | 3.29 | 1.00x | **1.63x** | 1.67x | 6.8e-6 | 5.5e-5 | 7.9e-4 | 1.6e-3 |
| 8192 | 128 | | 6.42 | 1.01x | **1.67x** | 1.67x | 6.7e-6 | 5.4e-5 | 7.8e-4 | 1.5e-3 |
| 8192 | 64 | yes | 1.75 | 0.98x | **1.42x** | 1.39x | 1.3e-5 | 1.0e-4 | 1.5e-3 | 3.0e-3 |
| 8192 | 128 | yes | 3.70 | 0.99x | **1.74x** | 1.72x | 1.3e-5 | 1.0e-4 | 1.5e-3 | 2.9e-3 |

Speedup is relative to FP16. FP8 reaches 143 TFLOPS at N=8192, D=128, against 86 for FP16.

- **BF16 buys nothing here.** Same HMMA rate as FP16, so the same speed, with 8x the
  error. That's 2^3, the gap between 7 and 10 mantissa bits. Its advantage is range,
  which attention inputs at this scale don't need.
- **FP8 is 1.3-1.7x, not 2x.** The matmuls do twice the work per instruction, but
  the softmax (max, exp, rescale) is the same amount of work. D=128 sits at
  1.6-1.75x throughout. D=64 has less matmul per softmax element and gets less: 1.3x
  at N=1024, 1.3-1.4x causal.
- **The error follows mantissa bits.** e4m3 (3 bits) is 116x FP16's error, against
  2^7 = 128. e5m2 (2 bits) is 228x, against 2^8 = 256. e5m2 is as fast as e4m3 and
  twice as wrong, so e4m3 is the one to use for the forward.
- **Most of the FP8 error is input rounding.** At N=4096, D=64, rounding Q/K/V to
  e4m3 and then computing in fp32 already gives 9.9e-4. The kernel adds another
  5.2e-4 on top. With FP16 inputs that in-kernel part is 5.7e-6, so it comes from
  casting P to FP8. Keeping P in FP16 could recover at most that smaller share.
- **Causal max error is large: 0.15 for e4m3.** The first rows attend to only a few
  keys, so their output is close to a single row of V, and V's rounding comes
  straight through without averaging out. Row 0's output is exactly `v[0]`.

Real activations have outliers, so a production FP8 path needs per-tensor or
per-block scales (FlashAttention-3 does this). None of that is here, and the backward
stays FP16.

```bash
python triton/precision_exp.py
```

## Triton vs CUDA: under the hood

Same shape for both, single head, `N=2048, D=64`, RTX 5070 Ti. Triton runs
`BLOCK_M=64, BLOCK_N=64, num_warps=4`; v4 runs `BR=64, BC=16, 4 warps`. Both
launch 32 CTAs of 128 threads, so the grid and the occupancy are identical and
the gap comes entirely from what each warp does per K/V tile.

| | CUDA v4 | Triton `num_stages=1` | Triton `num_stages=3` |
|---|--:|--:|--:|
| Wall-clock (CUDA events, 200 iters) | 0.457 ms | 0.044 ms | **0.037 ms** |
| Kernel time under ncu (locked clocks) | 606 us | 61 us | 39 us |
| Instructions executed | 14.1 M | 2.60 M | 2.60 M |
| Tensor pipe (HMMA) utilization | 2.2% | 22.6% | **35.3%** |
| Issue slots busy | 7.4% | 13.6% | 21.8% |
| Global load requests (same 532,480 sectors) | 266,240 | 33,280 | 33,280 |
| Shmem bank-conflict wavefronts / total | 19.4 M / 23.7 M | 0 / 0.77 M | 1 / 0.65 M |
| Registers / thread | 64 | 174 | 152 |
| Shared memory / block | 35 KB | 16 KB | 40 KB |
| Stall per issue: long scoreboard | 3.22 | 2.10 | 0.11 |
| Stall per issue: short scoreboard | 5.62 | 0.40 | 0.42 |
| Stall per issue: math pipe throttle | 0.11 | 1.41 | 1.43 |

Triton is ~12x faster than v4 on the same shape. Occupancy doesn't explain it
(8.3% achieved warps on all three), and neither does the Tensor Core instruction.
Everything else in the loop does.

### The mma is the same instruction

v4 calls `wmma::mma_sync` on 16x16x16 fragments, which PTX spells
`wmma.mma.sync.aligned.m16n16k16`. Triton's `tl.dot` emits
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` directly. ptxas lowers both
to the same SASS, `HMMA.16816.F32`: one wmma becomes two HMMAs. Operand loads
match too: `wmma::load_matrix_sync` from shmem and Triton's operand loads both
become `LDSM.16.M88.4` (`ldmatrix`), with `.MT88` for the transposed operand. The
TTGIR shows `#ttg.nvidia_mma<{versionMajor = 2}>`, the Ampere-style `mma.sync`
path. sm_120 doesn't have Hopper's `wgmma`, so there's no newer Tensor Core
instruction that Triton is using and v4 isn't.

What differs is how much work surrounds each HMMA. Inner loop only, normalized to
64 keys (four v4 iterations against one Triton iteration):

| Per 64 keys, per warp | CUDA v4 | Triton `num_stages=3` |
|---|--:|--:|
| SASS instructions | ~3,630 | ~620 |
| `HMMA.16816.F32` | 64 | 64 |
| `BAR.SYNC` | 12 | 2 |
| Global loads | 56x `LDG.E.U16` (2 B) | 8x `LDGSTS.E.BYPASS.128` (16 B, `cp.async`) |
| Shmem traffic outside `ldmatrix` | ~610 `LDS`/`STS` | 3 `LDS` |

HMMA is 1.8% of v4's loop and 10% of Triton's.

### Where v4's instructions go

**S and O live in shared memory.** v4 writes the `Q*K^T` fragment to `sS`, has 16
lanes read it back row by row for the softmax, writes `P` to `sP`, then each
`P*V` step loads the O accumulator from `sO`, runs one mma, and stores it back
(`attention.cuh:551`, `:590-594`). Triton never spills them: `qk`, `p` and `acc`
stay in registers in the mma accumulator layout for the whole loop. That costs
registers (152-174 against v4's 64) and saves the round-trip.

**The softmax is serial.** In v4, lane `r` owns row `r` and walks its 16 columns
in a loop, while lanes 16-31 idle (`attention.cuh:555`). In Triton's mma
layout each row is spread across a quad of 4 lanes, so a row max or row sum is
two `SHFL.BFLY` steps. That's 8 shuffles per tile for both reductions on both rows
a thread holds.

**Every shmem access conflicts.** 82% of v4's shmem wavefronts are bank-conflict
replays. Per-instruction source counters put them on every shmem opcode in the
loop: `LDS.128`/`STS.128` (the O fragment load/store, 21% + 19%), `LDSM` (the Q/K/P
fragment loads, 20% + 9.5%), and the scalar softmax `LDS`/`STS`. The cause is the
stride: `D=64` rows are 128 B of FP16 or 256 B of FP32, both multiples of the
128 B bank cycle, so every row of an 8x8 `ldmatrix` tile, and every lane in the
softmax `sO_row[d] *= alpha`, lands on the same banks. Triton's shared layout is
`#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8}>`: 16-byte chunks
XOR-permuted by `row % 8`, so it measures zero conflicts. That's the swizzle
v5 has to do by hand.

**Global loads are 2 bytes wide.** v4's cooperative K/V load indexes one `__half`
per thread per iteration, giving `LDG.E.U16`. Triton's `#blocked` layout gives
each thread 8 contiguous halves (`sizePerThread = [1, 8]`), so 16-byte loads. The
bytes moved are identical (532,480 sectors on both), but v4 issues 8x the
requests.

### What `num_stages` does

`num_stages=1` loads K/V with `LDG.E.128` into registers and `STS.128` into shmem,
the same two-step pattern v4 uses. From `num_stages=2` up, the TTGIR switches to
`ttg.async_copy_global_to_local` / `async_commit_group` / `async_wait`, which
becomes `cp.async.cg.shared.global` (`LDGSTS.E.BYPASS.128`, bypassing L1) with
`LDGDEPBAR` / `DEPBAR.LE` for the group waits. The prologue issues the first
tiles before the loop and the loop body issues tile `n+1` while computing tile
`n`. Shared memory goes from 16 KB to 40 KB for the extra buffers.

The effect shows up in the stalls: long-scoreboard (waiting on global data) goes
from 2.10 to 0.11 cycles per issue, and the Tensor pipe climbs from 22.6% to
35.3%. With stages=3 the top non-trivial stall becomes math-pipe throttle, so the
Tensor Cores are now the limiter. Wall-clock is 0.044 / 0.040 / 0.037 / 0.037 ms
for stages 1 / 2 / 3 / 4: the second buffer is most of the win and the fourth
adds nothing at this size.

### What this says about v5

v4's remaining gap isn't the Tensor Core instruction. It's the data movement
around it. In order of measured cost: keep S/P/O in registers instead of shmem,
swizzle the shmem tiles, widen global loads to 16 B, and pipeline them with
`cp.async`. Triton's TTGIR is a working spec for all four, written down by the
compiler.

Reproduce:

```bash
python triton/inspect_asm.py     # TTGIR / PTX / SASS for num_stages=1 and 3 -> docs/asm/
ncu --set full -k regex:_flash_fwd -c 2 python triton/inspect_asm.py --profile
ncu --set full -k regex:flash_attn_wmma_v4 -c 1 build\bin\attn.exe --N 2048 --D 64
```

`docs/asm/cuda_v4.{ptx,sass}` is the `flash_attn_wmma_v4_kernel<64,4>` entry cut
out of `nvcc -arch=sm_120 -ptx` / `cuobjdump -sass` on `src/main.cu`. The two ncu
reports are in `docs/profile/blackwell_*.ncu-rep`.

## CUDA v5

v5 starts as a copy of v4 and is changed one step at a time, each measured against
the previous one. v4 stays as it was, so the numbers above still reproduce.

### Heads in one launch

Q/K/V/O are contiguous `[B, H, N, D]`. The grid gets a second axis,
`blockIdx.y = b*H + h`, and each block offsets its four pointers by
`blockIdx.y * N * D`. Nothing else changes: a block still owns 64 query rows of
one head.

The comparison is v4 launched once per head vs v5 launched once (RTX 5070 Ti):

| shape | v4, one launch per head | v5, one launch | speedup |
|---|--:|--:|--:|
| H=3, N=1000, D=64 | 0.973 ms | 0.265 ms | 3.7x |
| H=16, N=2048, D=64 | 8.95 ms | 3.04 ms | 2.9x |
| B=2, H=4, N=1024, D=128 | 3.83 ms | 0.834 ms | 4.6x |

The kernel didn't get faster. The GPU got filled. One head at N=2048 is 32 blocks
of 64 rows, on a 70-SM GPU, so most SMs sit idle and every launch is a short
partial wave. Back-to-back launches on one stream don't overlap. At H=3, N=1000
the three heads are 48 blocks together, still under one per SM, and v5 takes
about as long as v4 takes for a single head (0.270 ms).

All heads match the CPU reference to the same error as v4 (max 5e-6), including
`N=1000`, which exercises the tail tile.

### Bank conflicts: padding first

A1 measured 82% of v4's shared wavefronts as conflict replays. Nsight's source
view splits that by line (`N=2048, D=64, H=16`, 310 M excess wavefronts):

| where | excess / total wavefronts |
|---|--:|
| softmax rescale, `sO_row[d] *= alpha` | 117 M / 134 M |
| wmma loads of Q, K, V, P (`LDSM`) | 92 M / 109 M |
| wmma accumulator load/store of S and O (`LDS.64` / `STS.64`) | 59 M / 80 M |
| softmax reads of S, writes of P | 48 M / 59 M |

The biggest one isn't a wmma access at all. In the rescale, each of 16 lanes owns
one row of O, and rows are 256 B apart, so all 16 lanes hit the same bank. Every
tile row stride is a multiple of 128 B, which is what makes the rest conflict.

XOR swizzling can't be used with these accesses. `wmma::load_matrix_sync` takes
a linear layout and a leading dimension, nothing else. So this step pads, and the
swizzle comes with hand-written `ldmatrix` next:

- **Rescale moved to the whole warp, `float4` per lane.** Adjacent lanes touch
  adjacent addresses. Alpha goes through a 64-float `sa` array.
- **Q/K/V rows padded to `D+8` halves** (144 B at D=64). The 8 rows an `ldmatrix`
  phase reads land on 8 different 16 B segments.
- **O rows padded to `D+8` floats.** With `D+4` the accumulator `LDS.64` /
  `STS.64` stayed 2-way in the source view (19 M excess). `D+8` removed it.
- **S rows padded to 20 floats.** S is written by the accumulator and read one row
  per lane, and those want different strides. 20 makes the row reads 2-way
  instead of 8-way.

Same occupancy as before: 41 KB/block at D=64 (2 blocks/SM), 70 KB at D=128
(1 block/SM).

| `N=2048, H=16` (ncu) | B1, D=64 | padded, D=64 | B1, D=128 | padded, D=128 |
|---|--:|--:|--:|--:|
| Bank-conflict wavefronts | 311 M | **10.7 M** | 570 M | **10.6 M** |
| Shared wavefronts total | 380 M | 82 M | 694 M | 140 M |
| Instructions | 245 M | 239 M | 349 M | 338 M |
| Stall/issue: short scoreboard | 8.17 | 2.52 | 6.88 | 2.56 |
| Stall/issue: long scoreboard | 1.00 | 2.27 | 2.88 | **5.34** |
| Kernel time | 2.85 ms | **1.73 ms** | 7.34 ms | **6.00 ms** |

Wall-clock against B1, same session:

| shape | B1 | padded | speedup |
|---|--:|--:|--:|
| H=3, N=1000, D=64 | 0.218 ms | 0.151 ms | 1.44x |
| H=16, N=2048, D=64 | 2.50 ms | 1.30 ms | 1.93x |
| H=16, N=4096, D=64 | 9.61 ms | 4.98 ms | 1.93x |
| H=16, N=2048, D=128 | 6.43 ms | 4.91 ms | 1.31x |
| H=16, N=4096, D=128 | 24.2 ms | 19.2 ms | 1.26x |

D=128 gains less, and the stall table says why. Shared-memory waits went down
(short scoreboard 6.9 to 2.6), but waits on global loads went up (long scoreboard
2.9 to 5.3). With one block per SM there are only 4 warps to hide the scalar
`LDG.E.U16` loads of K and V, so once the shared-memory stalls are gone the global
load latency is what's left exposed. Overlapping those loads with compute is the
job of `cp.async`.

An intermediate version did the rescale one float per lane. It removed the same
conflicts but executed 20-30% more instructions, so D=128 barely moved. `float4`
fixed that.

### mma.sync, registers, XOR swizzle

To swizzle, the shmem accesses have to be hand-written, and once wmma is gone
there's no reason to keep S, P and O in shmem either. So this step changes three
things together, the FA-2 structure:

- **`mma.sync.m16n8k16` and `ldmatrix` by hand** instead of wmma. It's the same
  `HMMA.16816.F32` underneath (A1).
- **S, P, O in registers.** Each warp keeps its 16 Q rows as A fragments for the
  whole loop. S comes out of the mma in accumulator layout. Row max and row sum
  reduce across the 4 lanes that share a row with two `shfl.xor`. The accumulator
  layout of S n-tiles `2k, 2k+1` is exactly the A-fragment layout for k-step `k`,
  so P is packed to `half2` and fed straight into `O += P V`. V is read with
  `ldmatrix.trans`. sS, sP, sO and the `sa` array are gone. Shared memory is only
  Q, K, V.
- **XOR swizzle on Q/K/V.** A row is `D/8` chunks of 16 B, and chunk `c` of row
  `r` is stored at `c ^ (r % 8)`. An `ldmatrix` phase reads the same logical chunk
  from 8 consecutive rows, which now land on 8 different bank groups without any
  padding.

The global loads stay scalar `LDG.E.U16` so that B3 changes only the loads. The
K/V tile `BC` is a template parameter. `BC=16` is v4's tile, now on registers.

ncu kernel time, H=16, best of 3. The GPU drifted between 0.9 and 2.3 GHz even
with `--clock-control base`, so every row is compared at matching clocks, and the
N=4096, D=128 row (1.9-2.1 GHz) is in cycles:

| shape | padded (B2a) | BC=16 | BC=32 | BC=64 | vs B2a |
|---|--:|--:|--:|--:|--:|
| N=2048, D=64 | 1.83 ms | 0.549 | 0.386 | **0.361** | 5.1x |
| N=4096, D=64 | 6.34 ms | 1.72 | **1.32** | **1.32** | 4.8x |
| N=2048, D=128 | 6.85 ms | 1.64 | **0.957** | 1.23 | 7.2x |
| N=4096, D=128 | 51.7 M cyc | 12.5 M | **7.29 M** | 8.71 M | 7.1x |

| `N=2048, H=16` (ncu) | B2a D=64 | v5 D=64, BC=64 | B2a D=128 | v5 D=128, BC=32 |
|---|--:|--:|--:|--:|
| Bank-conflict wavefronts | 10.7 M | 0.21 M | 10.6 M | 0.28 M |
| Instructions | 239 M | 113 M | 338 M | 202 M |
| Registers / thread | 48 | 128 | 60 | 165 |
| Achieved warps | 15.8% | 30.4% | 8.3% | 22.0% |
| Issue slots busy | 21.9% | 51.1% | 8.6% | 34.8% |
| Stall/issue: short scoreboard | 2.51 | 0.09 | 2.56 | 0.13 |
| Stall/issue: long scoreboard | 2.12 | 2.40 | 5.18 | 3.23 |

- **This step's speedup isn't attributable to the swizzle alone.** Most of it is
  what's no longer done: no S/P/O stores and reloads, no serial 16-lane softmax,
  no rescale pass. Instructions drop 53% at D=64 and 40% at D=128. Padding had
  already removed 97% of the
  conflicts. The swizzle's job here is to keep the last ones out without padding,
  now that every shmem access is an `ldmatrix`.
- **Occupancy went up, not down,** despite 128-231 registers per thread. Shared
  memory per block fell from 41/70 KB to 24/32 KB (`(64 + 2*BC) * D * 2 B`).
- **BC trades loop overhead against registers.** BC=16 pays one `__syncthreads`
  pair and one softmax pass per 16 keys. BC=64 at D=128 needs 231 registers,
  which allows only 2 blocks per SM, so BC=32 (165 registers, 3 blocks) wins there.
  The default is BC=64 for D=64 and BC=32 for D=128. No spills in any config.
- **What's left is global loads.** Shared-memory stalls are gone (short
  scoreboard ~0.1). Long scoreboard is now the largest stall, and the loads are
  still one half per thread per instruction, issued and then waited on before any
  compute starts.

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
build\bin\attn.exe --N 2048 --D 64 --B 1 --H 16
build\bin\attn.exe --N 2048 --D 128 --H 16 --BC 64     REM v5 K/V tile, default 64 (D=64) / 32 (D=128)
```

All GPU kernels run, check against a CPU double-precision reference, and print
per-call timings. naive through v4 run on head 0; v5 runs every head.

PyTorch reference numbers:
```bash
python scripts\torch_bench.py
```

## Files

```
src/
  attention.cuh   naive + flash v1 + v2 (Q-tile) + v3 (wmma BR=16) + v4 (wmma BR=64)
                  + v5 ([B, H, N, D], mma.sync + ldmatrix, S/P/O in registers, swizzled shmem)
  main.cu         harness: CPU ref, correctness diff, timing, FP32->FP16 conv kernel
scripts/
  torch_bench.py  PyTorch SDPA reference (FP32 + FP16)
triton/
  flash_triton.py Triton port, multi-head, validated vs torch SDPA
  inspect_asm.py  dumps Triton TTGIR/PTX/SASS; --profile does one launch for ncu
  autotune_sweep.py  times all 24 autotune configs per shape -> docs/autotune_sweep.csv
  flash_bwd.py    backward: naive (materialized) and tiled (recompute from lse)
  test_bwd.py     gradcheck (fp64) + fp16 grads vs fp32 reference
  bwd_bench.py    backward timing + memory: naive vs tiled vs SDPA
  precision_exp.py  forward speed + error: FP16 vs BF16 vs FP8 (e4m3, e5m2)
docs/profile/
  flash.ncu-rep / qtile.ncu-rep / wmma.ncu-rep / wmma_v4.ncu-rep   Nsight Compute, v1-v4 (Ada)
  blackwell_v4.ncu-rep / blackwell_triton.ncu-rep                  v4 vs Triton, --set full (Blackwell)
docs/asm/
  triton_s{1,3}.{ttgir,ptx,sass}   Triton output at num_stages=1 and 3
  cuda_v4.{ptx,sass}               v4<64,4> from nvcc / cuobjdump
```

## Scope and limitations

A deliberately narrow study kernel, kept small so each optimization is isolated and
measurable:

- **Backward is Triton-only.** The CUDA kernels are forward only.
- **Non-causal** on the CUDA side, and single head up to v4 (v5 takes
  `[B, H, N, D]`). The Triton port is multi-head
  (`[Z, H, N, D]`) with an optional causal mask and tile skip.
- **`D` in {64, 128}, a multiple of 16** to match the wmma tile width. Other head
  dimensions are unsupported on the wmma paths.
- **FP16 inputs on v3/v4**, FP32 accumulate. FP8 is Triton forward only, unscaled.
- **Single GPU.** No multi-GPU or distributed.

The goal was to take one kernel from a correct baseline to ~4x by profiling one
bottleneck at a time, not to reimplement production FlashAttention-2.

## Hardware

Developed on RTX 4080 Laptop (Ada, SM 8.9, 58 SMs, 12 GB, ~40 MB L2, 100 KB
shmem/SM). Re-benchmarked on RTX 5070 Ti (Blackwell, SM 12.0).
