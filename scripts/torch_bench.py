"""torch_bench.py  —  timing reference for PyTorch's scaled_dot_product_attention.

PyTorch uses FlashAttention-2 (FP16, Tensor Cores) on Ada GPUs automatically.
Included here as a ceiling number to contextualise my FP32 implementations.
"""

import argparse
import time

import torch
import torch.nn.functional as F


def bench(N, D, device="cuda", dtype=torch.float32, iters=20, warmup=5):
    Q = torch.randn(1, 1, N, D, device=device, dtype=dtype)
    K = torch.randn(1, 1, N, D, device=device, dtype=dtype)
    V = torch.randn(1, 1, N, D, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"{'N':>6}  {'D':>3}  {'fp32 ms':>10}  {'fp16 ms':>10}")
    for N in (512, 1024, 2048, 4096):
        for D in (64, 128):
            t_fp32 = bench(N, D, dtype=torch.float32, iters=args.iters)
            t_fp16 = bench(N, D, dtype=torch.float16, iters=args.iters)
            print(f"{N:>6}  {D:>3}  {t_fp32:>10.3f}  {t_fp16:>10.3f}")


if __name__ == "__main__":
    main()
