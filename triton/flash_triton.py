# Triton port of the FlashAttention forward pass.
# Same online-softmax tiling as the CUDA version (attention.cuh), but Triton
# handles the warp/shmem layout and the wmma lowering. Non-causal, multi-head.
# Layout: Q/K/V are [Z, H, N, D]. Checked against torch SDPA.

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd(Q, K, V, sm_scale, Out,
               sqz, sqh, sqm, sqk,
               skz, skh, skn, skk,
               svz, svh, svn, svk,
               soz, soh, som, sok,
               H, N,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)
    off_z   = off_hz // H
    off_h   = off_hz % H

    q_base = Q + off_z * sqz + off_h * sqh
    k_base = K + off_z * skz + off_h * skh
    v_base = V + off_z * svz + off_h * svh
    o_base = Out + off_z * soz + off_h * soh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # 这个 block 负责的 Q 分块，一次性读进来
    q = tl.load(q_base + offs_m[:, None] * sqm + offs_d[None, :] * sqk,
                mask=offs_m[:, None] < N, other=0.0)

    # 在线 softmax 的运行状态：行最大值 m、归一化和 l、输出累加器 acc
    m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # 沿 K/V 分块流式扫描
    for start_n in range(0, N, BLOCK_N):
        n_idx = start_n + offs_n
        k = tl.load(k_base + n_idx[:, None] * skn + offs_d[None, :] * skk,
                    mask=n_idx[:, None] < N, other=0.0)
        # S = Q * K^T，Tensor Core 由 tl.dot 自动下放
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(n_idx[None, :] < N, qk, -float('inf'))   # 屏蔽尾块越界列

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)            # 旧累加器的 rescale 因子
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + n_idx[:, None] * svn + offs_d[None, :] * svk,
                    mask=n_idx[:, None] < N, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)        # O += P * V
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * som + offs_d[None, :] * sok,
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


def flash_attn(q, k, v, BLOCK_M=64, BLOCK_N=64):
    Z, H, N, D = q.shape
    sm_scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), Z * H)
    _flash_fwd[grid](q, k, v, sm_scale, o,
                     *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                     H, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D)
    return o


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


if __name__ == "__main__":
    import torch.nn.functional as F
    from torch.backends.cuda import flash_sdp_enabled
    torch.manual_seed(42)
    print(torch.cuda.get_device_name(0), "triton", triton.__version__, "torch", torch.__version__)
    # On this Windows torch build the FLASH/cuDNN backends are NOT compiled in,
    # so SDPA falls back to the mem-efficient kernel. The timing below compares
    # against THAT, not against production FlashAttention. Do not call it
    # "faster than FlashAttention" - to do that, benchmark on Linux + flash-attn.
    print("NOTE: torch SDPA here = mem-efficient backend (flash backend unavailable on this build)")
    print(f"{'shape':>20} {'max|d| vs SDPA':>16} {'triton ms':>10} {'SDPA(mem-eff) ms':>18}")
    for (Z, H, N, D) in [(1, 16, 2048, 64), (1, 16, 4096, 64), (1, 16, 4096, 128)]:
        q = torch.randn(Z, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(Z, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(Z, H, N, D, device='cuda', dtype=torch.float16)
        o_tri = flash_attn(q, k, v)
        o_ref = F.scaled_dot_product_attention(q, k, v)   # non-causal
        err = (o_tri - o_ref).abs().max().item()
        t_tri = _bench(lambda: flash_attn(q, k, v))
        t_ref = _bench(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"Z{Z} H{H} N{N} D{D}".rjust(20)
              + f"{err:>16.3e}{t_tri:>10.3f}{t_ref:>10.3f}")
    print("TRITON_FLASH_OK")
