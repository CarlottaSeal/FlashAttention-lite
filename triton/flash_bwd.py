# FlashAttention backward in Triton.
# naive: materializes S/P/dP/dS as [B, N, N], one kernel per step. Reference for the tiled version.
import torch
import triton
import triton.language as tl


@triton.jit
def _bmm(A, B, C, M, N, K,
         sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, alpha,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # C[b] = alpha * A[b] @ B[b]；转置靠传换过的 stride
    pid_m, pid_n, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ty = C.dtype.element_ty
    acc = tl.zeros([BM, BN], ty)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(A + b * sab + rm[:, None] * sam + kk[None, :] * sak,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        bb = tl.load(B + b * sbb + kk[:, None] * sbk + rn[None, :] * sbn,
                     mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a.to(ty), bb.to(ty), input_precision="ieee")
    tl.store(C + b * scb + rm[:, None] * scm + rn[None, :] * scn, acc * alpha,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _p_from_s(S, LSE, N, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    # 原地 S -> P = exp(S - lse)，causal 时上三角置 0
    row, b = tl.program_id(0), tl.program_id(1)
    base = S + (b * N + row) * N
    lse = tl.load(LSE + b * N + row)
    for c0 in range(0, N, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        s = tl.load(base + c, mask=c < N)
        p = tl.exp(s - lse)
        if CAUSAL:
            p = tl.where(c <= row, p, 0.0)
        tl.store(base + c, p, mask=c < N)


@triton.jit
def _ds_from_dp(P, DP, N, BLOCK: tl.constexpr):
    # 原地 dP -> dS = P * (dP - rowsum(P * dP))
    row, b = tl.program_id(0), tl.program_id(1)
    off = (b * N + row) * N
    delta = tl.zeros([BLOCK], DP.dtype.element_ty)
    for c0 in range(0, N, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        delta += tl.load(P + off + c, mask=c < N, other=0.0) * tl.load(DP + off + c, mask=c < N, other=0.0)
    d = tl.sum(delta, 0)
    for c0 in range(0, N, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        p = tl.load(P + off + c, mask=c < N)
        dp = tl.load(DP + off + c, mask=c < N)
        tl.store(DP + off + c, p * (dp - d), mask=c < N)


def _mm(a, b, c, alpha=1.0, ta=False, tb=False):
    # c = alpha * op(a) @ op(b)，全是 [B, rows, cols] 连续张量
    B = a.shape[0]
    sab, sam, sak = a.stride()
    sbb, sbk, sbn = b.stride()
    if ta: sam, sak = sak, sam
    if tb: sbk, sbn = sbn, sbk
    M, N = c.shape[1], c.shape[2]
    K = a.shape[1] if ta else a.shape[2]
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64), B)
    _bmm[grid](a, b, c, M, N, K, sab, sam, sak, sbb, sbk, sbn, *c.stride(), alpha,
               BM=64, BN=64, BK=32)
    return c


def flash_bwd_naive(do, q, k, v, o, lse, causal):
    Z, H, N, D = q.shape
    B = Z * H
    scale = D ** -0.5
    acc_ty = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, do = (t.reshape(B, N, D).contiguous() for t in (q, k, v, do))
    lse = lse.reshape(B, N).contiguous()

    P = _mm(q, k, torch.empty(B, N, N, device=q.device, dtype=acc_ty), scale, tb=True)
    _p_from_s[(N, B)](P, lse, N, CAUSAL=causal, BLOCK=256)

    dv = _mm(P, do, torch.empty(B, N, D, device=q.device, dtype=acc_ty), ta=True)
    dS = _mm(do, v, torch.empty(B, N, N, device=q.device, dtype=acc_ty), tb=True)
    _ds_from_dp[(N, B)](P, dS, N, BLOCK=256)

    dq = _mm(dS, k, torch.empty(B, N, D, device=q.device, dtype=acc_ty), scale)
    dk = _mm(dS, q, torch.empty(B, N, D, device=q.device, dtype=acc_ty), scale, ta=True)
    return tuple(t.reshape(Z, H, N, D).to(q.dtype) for t in (dq, dk, dv))
