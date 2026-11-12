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


# ---- tiled: 不物化 N x N，用 forward 存的 lse 逐块重算 P ----

@triton.jit
def _delta(O, DO, Delta, N, D: tl.constexpr, BLOCK: tl.constexpr):
    # Delta_i = rowsum(dO_i * O_i)，等于 rowsum(P_i * dP_i)，但只要 O(N*D)
    pid, b = tl.program_id(0), tl.program_id(1)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    off = (b * N + r[:, None]) * D + d[None, :]
    o = tl.load(O + off, mask=r[:, None] < N, other=0.0).to(Delta.dtype.element_ty)
    do = tl.load(DO + off, mask=r[:, None] < N, other=0.0).to(Delta.dtype.element_ty)
    tl.store(Delta + b * N + r, tl.sum(o * do, 1), mask=r < N)


@triton.jit
def _dkdv_inner(dk, dv, k, v, q_base, do_base, lse_base, d_base, scale, N,
                offs_n, offs_d, lo, hi, BLOCK_M: tl.constexpr, D: tl.constexpr,
                MASK: tl.constexpr):
    # 固定一块 K/V，扫 Q 块 [lo, hi)；全程按转置算，P^T 是 [BLOCK_N, BLOCK_M]
    for start_m in range(lo, hi, BLOCK_M):
        m = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(q_base + m[:, None] * D + offs_d[None, :], mask=m[:, None] < N, other=0.0)
        do = tl.load(do_base + m[:, None] * D + offs_d[None, :], mask=m[:, None] < N, other=0.0)
        lse = tl.load(lse_base + m, mask=m < N, other=float('inf'))   # 越界行 P=0
        delta = tl.load(d_base + m, mask=m < N, other=0.0)

        pT = tl.exp(tl.dot(k, tl.trans(q)) * scale - lse[None, :])
        if MASK:
            pT = tl.where(m[None, :] >= offs_n[:, None], pT, 0.0)
        dv += tl.dot(pT.to(do.dtype), do)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk += tl.dot(dsT.to(q.dtype), q)
    return dk, dv


@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, Delta, DK, DV, scale, N,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
              CAUSAL: tl.constexpr):
    pid, b = tl.program_id(0), tl.program_id(1)
    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    base = b * N * D
    kv_off = base + offs_n[:, None] * D + offs_d[None, :]
    k = tl.load(K + kv_off, mask=offs_n[:, None] < N, other=0.0)
    v = tl.load(V + kv_off, mask=offs_n[:, None] < N, other=0.0)
    ACC: tl.constexpr = tl.float64 if Q.dtype.element_ty == tl.float64 else tl.float32
    dk = tl.zeros([BLOCK_N, D], ACC)
    dv = tl.zeros([BLOCK_N, D], ACC)

    if CAUSAL:
        # 只有 i >= j 的 Q 行对这块 K/V 有贡献：对角块带 mask，下面的块整块可见，上面的跳过
        dk, dv = _dkdv_inner(dk, dv, k, v, Q + base, DO + base, LSE + b * N, Delta + b * N,
                             scale, N, offs_n, offs_d, start_n, start_n + BLOCK_N,
                             BLOCK_M, D, True)
        dk, dv = _dkdv_inner(dk, dv, k, v, Q + base, DO + base, LSE + b * N, Delta + b * N,
                             scale, N, offs_n, offs_d, start_n + BLOCK_N, N,
                             BLOCK_M, D, False)
    else:
        dk, dv = _dkdv_inner(dk, dv, k, v, Q + base, DO + base, LSE + b * N, Delta + b * N,
                             scale, N, offs_n, offs_d, 0, N, BLOCK_M, D, False)

    tl.store(DK + kv_off, (dk * scale).to(DK.dtype.element_ty), mask=offs_n[:, None] < N)
    tl.store(DV + kv_off, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < N)


@triton.jit
def _dq_inner(dq, q, do, lse, delta, k_base, v_base, scale, N, offs_m, offs_d, lo, hi,
              BLOCK_N: tl.constexpr, D: tl.constexpr, MASK: tl.constexpr):
    # 固定一块 Q，扫 K/V 块 [lo, hi)
    for start_n in range(lo, hi, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + n[:, None] * D + offs_d[None, :], mask=n[:, None] < N, other=0.0)
        v = tl.load(v_base + n[:, None] * D + offs_d[None, :], mask=n[:, None] < N, other=0.0)
        p = tl.exp(tl.dot(q, tl.trans(k)) * scale - lse[:, None])
        p = tl.where(n[None, :] < N, p, 0.0)
        if MASK:
            p = tl.where(offs_m[:, None] >= n[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    return dq


@triton.jit
def _bwd_dq(Q, K, V, DO, LSE, Delta, DQ, scale, N,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
            CAUSAL: tl.constexpr):
    pid, b = tl.program_id(0), tl.program_id(1)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = b * N * D
    q_off = base + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(Q + q_off, mask=offs_m[:, None] < N, other=0.0)
    do = tl.load(DO + q_off, mask=offs_m[:, None] < N, other=0.0)
    lse = tl.load(LSE + b * N + offs_m, mask=offs_m < N, other=float('inf'))
    delta = tl.load(Delta + b * N + offs_m, mask=offs_m < N, other=0.0)
    ACC: tl.constexpr = tl.float64 if Q.dtype.element_ty == tl.float64 else tl.float32
    dq = tl.zeros([BLOCK_M, D], ACC)

    if CAUSAL:
        dq = _dq_inner(dq, q, do, lse, delta, K + base, V + base, scale, N, offs_m, offs_d,
                       0, start_m, BLOCK_N, D, False)
        dq = _dq_inner(dq, q, do, lse, delta, K + base, V + base, scale, N, offs_m, offs_d,
                       start_m, start_m + BLOCK_M, BLOCK_N, D, True)
    else:
        dq = _dq_inner(dq, q, do, lse, delta, K + base, V + base, scale, N, offs_m, offs_d,
                       0, N, BLOCK_N, D, False)

    tl.store(DQ + q_off, (dq * scale).to(DQ.dtype.element_ty), mask=offs_m[:, None] < N)


def flash_bwd(do, q, k, v, o, lse, causal, BLOCK=64, num_stages=None):
    Z, H, N, D = q.shape
    if num_stages is None:
        num_stages = 3 if D <= 64 else 2   # D=128 三级超 99 KB
    B = Z * H
    q, k, v, o, do = (t.contiguous() for t in (q, k, v, o, do))
    delta = torch.empty_like(lse)
    _delta[(triton.cdiv(N, BLOCK), B)](o, do, delta, N, D=D, BLOCK=BLOCK)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    grid = (triton.cdiv(N, BLOCK), B)
    _bwd_dkdv[grid](q, k, v, do, lse, delta, dk, dv, D ** -0.5, N,
                    BLOCK_M=BLOCK, BLOCK_N=BLOCK, D=D, CAUSAL=causal,
                    num_stages=num_stages)
    _bwd_dq[grid](q, k, v, do, lse, delta, dq, D ** -0.5, N,
                  BLOCK_M=BLOCK, BLOCK_N=BLOCK, D=D, CAUSAL=causal,
                    num_stages=num_stages)
    return dq, dk, dv
