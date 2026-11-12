# Time every autotune config on every shape, record regs/smem/spills -> docs/autotune_sweep.csv.
# Then check flash_attn_tuned picks the same winner.
import csv
import os
import torch
import triton
from triton.testing import do_bench
from flash_triton import _flash_fwd, _flash_fwd_tuned, CONFIGS, flash_attn_tuned

Z, H = 1, 16
SHAPES = [(n, d) for d in (64, 128) for n in (1024, 2048, 4096, 8192)]


def run(cfg, q, k, v, o, causal):
    bm, bn = cfg.kwargs['BLOCK_M'], cfg.kwargs['BLOCK_N']
    N, D = q.shape[2], q.shape[3]
    return _flash_fwd[(triton.cdiv(N, bm), Z * H)](
        q, k, v, D ** -0.5, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, N, BLOCK_M=bm, BLOCK_N=bn, D=D, CAUSAL=causal,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'autotune_sweep.csv')
    rows = []
    for causal in (False, True):
        for N, D in SHAPES:
            q, k, v = (torch.randn(Z, H, N, D, device='cuda', dtype=torch.float16) for _ in range(3))
            o = torch.empty_like(q)
            res = []
            for cfg in CONFIGS:
                c = cfg.kwargs
                try:
                    kern = run(cfg, q, k, v, o, causal)
                    ms = do_bench(lambda: run(cfg, q, k, v, o, causal))
                    regs, spills, smem = kern.n_regs, kern.n_spills, kern.metadata.shared
                except triton.runtime.errors.OutOfResources:
                    ms, regs, spills, smem = float('inf'), '', '', 'OOR'
                res.append((ms, c['BLOCK_M'], c['BLOCK_N'], cfg.num_warps, cfg.num_stages, regs, spills, smem))
            best = min(r[0] for r in res)
            for r in res:
                rows.append([int(causal), N, D, *r[1:], f"{r[0]:.4f}", f"{r[0] / best:.3f}"])
            w = min(res)
            flash_attn_tuned(q, k, v, causal)
            print(f"causal={int(causal)} N={N:5} D={D:3} best {w[0]:.4f} ms "
                  f"BM={w[1]} BN={w[2]} w={w[3]} s={w[4]} regs={w[5]} smem={w[7]} "
                  f"| autotune picked {_flash_fwd_tuned.best_config}")
    with open(out, 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['causal', 'N', 'D', 'BLOCK_M', 'BLOCK_N', 'num_warps', 'num_stages',
                     'regs', 'spills', 'smem_B', 'ms', 'vs_best'])
        cw.writerows(rows)
