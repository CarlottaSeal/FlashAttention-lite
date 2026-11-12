# Backward only: naive (materialized) vs tiled vs torch SDPA (mem-efficient backend on this build).
import torch
import torch.nn.functional as F
from triton.testing import do_bench
from flash_triton import flash_attn
from flash_bwd import flash_bwd, flash_bwd_naive
from test_bwd import ref_attn

Z, H = 1, 16


def peak_mb(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"{'N':>5} {'D':>4} {'causal':>6} | {'naive ms':>9} {'tiled ms':>9} {'SDPA ms':>8} | "
          f"{'naive MB':>9} {'tiled MB':>9} {'SDPA MB':>8}")
    for causal in (False, True):
        for D in (64, 128):
            for N in (1024, 2048, 4096, 8192):
                q, k, v = (torch.randn(Z, H, N, D, device='cuda', dtype=torch.float16) for _ in range(3))
                do = torch.randn_like(q)
                o, lse = flash_attn(q, k, v, causal, return_lse=True)

                tiled = lambda: flash_bwd(do, q, k, v, o, lse, causal)
                naive = lambda: flash_bwd_naive(do, q, k, v, o, lse, causal)
                qs, ks, vs = (t.clone().requires_grad_() for t in (q, k, v))
                os_ = F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal)
                sdpa = lambda: torch.autograd.grad(os_, (qs, ks, vs), do, retain_graph=True)

                fit = N <= 4096   # naive 的 4 个 [B,N,N] fp32 在 N=8192 要 16 GB
                t_n = f"{do_bench(naive):9.3f}" if fit else f"{'-':>9}"
                m_n = f"{peak_mb(naive):9.0f}" if fit else f"{'-':>9}"
                print(f"{N:5} {D:4} {int(causal):6} | {t_n} {do_bench(tiled):9.3f} {do_bench(sdpa):8.3f} | "
                      f"{m_n} {peak_mb(tiled):9.0f} {peak_mb(sdpa):8.0f}")
                del os_

    # fp16 误差对照：同一个 fp32 参考，比较 tiled 和 SDPA 自己的 fp16 backward
    for causal in (False, True):
        q, k, v = (torch.randn(2, 4, 1024, 64, device='cuda', dtype=torch.float16) for _ in range(3))
        do = torch.randn_like(q)
        qr, kr, vr = (t.float().requires_grad_() for t in (q, k, v))
        ref_attn(qr, kr, vr, causal).backward(do.float())
        o, lse = flash_attn(q, k, v, causal, return_lse=True)
        mine = flash_bwd(do, q, k, v, o, lse, causal)
        qs, ks, vs = (t.clone().requires_grad_() for t in (q, k, v))
        theirs = torch.autograd.grad(F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal), (qs, ks, vs), do)
        e = lambda gs: " ".join(f"{(g.float() - r.grad).abs().max().item():.2e}" for g, r in zip(gs, (qr, kr, vr)))
        print(f"causal={int(causal)} max|d| vs fp32 ref (dq dk dv): tiled {e(mine)} | SDPA {e(theirs)}")
