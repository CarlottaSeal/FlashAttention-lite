# Dump what Triton generated for _flash_fwd (TTGIR / PTX / SASS) and run one
# launch for ncu. Shape matches the CUDA Nsight table: single head, N=2048, D=64.
#   python triton/inspect_asm.py                 # dump to docs/asm/
#   python triton/inspect_asm.py --profile       # one launch, for ncu
import os
import sys
import torch
from flash_triton import _flash_fwd

N, D = 2048, 64
BLOCK_M, BLOCK_N = 64, 64


def launch(num_stages):
    q, k, v = (torch.randn(1, 1, N, D, device='cuda', dtype=torch.float16) for _ in range(3))
    o = torch.empty_like(q)
    lse = torch.empty(N, device='cuda')
    grid = ((N + BLOCK_M - 1) // BLOCK_M, 1)
    kern = _flash_fwd[grid](q, k, v, 1.0 / D ** 0.5, o, lse,
                            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                            1, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
                            num_warps=4, num_stages=num_stages)
    torch.cuda.synchronize()
    return kern


if __name__ == "__main__":
    stages = [int(a.split('=')[1]) for a in sys.argv if a.startswith('--stages=')] or [1, 3]
    if '--profile' in sys.argv:
        for s in stages:
            launch(s)
        sys.exit(0)

    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'asm')
    os.makedirs(out, exist_ok=True)
    for s in stages:
        kern = launch(s)
        for ir in ('ttgir', 'ptx', 'sass'):
            with open(os.path.join(out, f'triton_s{s}.{ir}'), 'w') as f:
                f.write(kern.asm[ir])
        md = kern.metadata
        print(f"num_stages={s}: shared={md.shared} B, num_warps={md.num_warps}")
