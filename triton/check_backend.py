import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
q = torch.randn(1, 16, 4096, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q); v = torch.randn_like(q)

def bench(fn, it=50, wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it

backends = {
    "FLASH": SDPBackend.FLASH_ATTENTION,
    "MEM_EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "MATH": SDPBackend.MATH,
}
print(torch.cuda.get_device_name(0), "torch", torch.__version__)
print("default (no context):", end=" ")
try:
    print(f"{bench(lambda: F.scaled_dot_product_attention(q,k,v)):.3f} ms")
except Exception as e:
    print("ERR", e)

for name, be in backends.items():
    try:
        with sdpa_kernel(be):
            t = bench(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"{name:>14}: {t:.3f} ms")
    except Exception as e:
        print(f"{name:>14}: unavailable ({type(e).__name__})")
