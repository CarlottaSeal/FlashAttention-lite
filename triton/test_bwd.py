# gradcheck in fp64 on small shapes, then fp16 grads vs an fp32 PyTorch reference.
import torch
from flash_triton import attention


def ref_attn(q, k, v, causal):
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        n = q.shape[2]
        s = s.masked_fill(torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1), float('-inf'))
    return torch.softmax(s, -1) @ v


def rand(*shape, dtype, grad=True):
    return torch.randn(*shape, device='cuda', dtype=dtype).requires_grad_(grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    # D=16 时 scale=0.25 精确；N=40/100 走尾块 mask
    for bwd in ('naive', 'flash'):
        for causal in (False, True):
            for N in (40, 100):
                q, k, v = (rand(1, 2, N, 16, dtype=torch.float64) for _ in range(3))
                ok = torch.autograd.gradcheck(lambda q, k, v: attention(q, k, v, causal, bwd), (q, k, v))
                print(f"gradcheck fp64 {bwd} causal={causal} N={N}: {ok}")

    for bwd, causal in [(b, c) for b in ('naive', 'flash') for c in (False, True)]:
        for N, D in ((1024, 64), (2048, 128), (1000, 64)):
            q, k, v = (rand(2, 4, N, D, dtype=torch.float16) for _ in range(3))
            do = torch.randn_like(q)
            attention(q, k, v, causal, bwd).backward(do)
            got = [t.grad.clone() for t in (q, k, v)]
            qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
            ref_attn(qr, kr, vr, causal).backward(do.float())
            errs = [(g.float() - r.grad).abs().max().item() for g, r in zip(got, (qr, kr, vr))]
            print(f"fp16 {bwd:5} causal={int(causal)} N={N} D={D}  max|d| dq={errs[0]:.2e} dk={errs[1]:.2e} dv={errs[2]:.2e}")
    print("BWD_OK")
