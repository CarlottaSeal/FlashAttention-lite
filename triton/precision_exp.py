# Forward precision vs speed: FP16 / BF16 / FP8 (e4m3, e5m2) inputs, same kernel.
# Error is against fp32 SDPA on the original fp32 inputs, so it includes input rounding.
import torch
import torch.nn.functional as F
from triton.testing import do_bench
from flash_triton import flash_attn

Z, H = 1, 16
DTYPES = [('fp16', torch.float16), ('bf16', torch.bfloat16),
          ('e4m3', torch.float8_e4m3fn), ('e5m2', torch.float8_e5m2)]
TILES = [(64, 64, 4, 3), (128, 64, 8, 3), (128, 32, 4, 2)]   # 每种 dtype 取最快的


def best_ms(q, k, v, causal):
    ts = []
    for bm, bn, w, s in TILES:
        try:
            ts.append(do_bench(lambda: flash_attn(q, k, v, causal, BLOCK_M=bm, BLOCK_N=bn,
                                                  num_warps=w, num_stages=s)))
        except Exception:   # OutOfResources
            pass
    return min(ts)


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"{'N':>5} {'D':>4} {'c':>2} {'dtype':>5} | {'ms':>7} {'TFLOPS':>7} {'x fp16':>7} | {'max|d|':>9} {'mean|d|':>9}")
    for causal in (False, True):
        for D in (64, 128):
            for N in (1024, 4096, 8192):
                x = [torch.randn(Z, H, N, D, device='cuda') for _ in range(3)]
                ref = F.scaled_dot_product_attention(*x, is_causal=causal)
                flops = 4 * Z * H * N * N * D * (0.5 if causal else 1)
                base = None
                for name, dt in DTYPES:
                    q, k, v = (t.to(dt) for t in x)
                    d = (flash_attn(q, k, v, causal).float() - ref).abs()
                    ms = best_ms(q, k, v, causal)
                    base = base or ms
                    print(f"{N:5} {D:4} {int(causal):2} {name:>5} | {ms:7.3f} {flops / ms * 1e-9:7.1f} "
                          f"{base / ms:7.2f} | {d.max().item():9.2e} {d.mean().item():9.2e}")

    # 拆误差：输入舍入（fp32 算舍入后的输入）vs kernel 内部（P 也降到输入精度）
    print(f"\nN=4096 D=64 non-causal, mean|d| split: {'input rounding':>15} {'in-kernel':>10} {'total':>9}")
    x = [torch.randn(Z, H, 4096, 64, device='cuda') for _ in range(3)]
    ref = F.scaled_dot_product_attention(*x)
    for name, dt in DTYPES:
        q, k, v = (t.to(dt) for t in x)
        ref_r = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        o = flash_attn(q, k, v).float()
        m = lambda a, b: (a - b).abs().mean().item()
        print(f"{name:>5} {m(ref_r, ref):>50.2e} {m(o, ref_r):>10.2e} {m(o, ref):>9.2e}")
