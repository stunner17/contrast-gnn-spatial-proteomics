"""Fused Triton kernel for the contrast operator  c_i = mean_{j in N(i)} |h_i - h_j|.

Reference (PyTorch Geometric, as in models.py):
    scatter((h[dst] - h[src]).abs(), dst, reduce="mean")
which materialises an E x F tensor of differences (6.4 M edges x F floats on METABRIC) before reducing.

The fused kernel walks a CSR adjacency (rows = destination nodes, sorted), keeps the running sum in
registers and writes only the N x F output, so the E x F intermediate is never created.

Backward, for a SYMMETRIC graph (every edge j->i has i->j; true for our tissue graphs, checked by
build_csr):  dL/dh_k = sum_{j in N(k)} sign(h_k - h_j) * (g_k / deg_k + g_j / deg_j)
This is a second gather over the same CSR, again with no atomics and no E x F tensor.
sign(0) = 0, matching torch's abs() gradient. Nodes with no neighbours give 0, matching scatter-mean.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _contrast_fwd(h_ptr, rowptr_ptr, col_ptr, out_ptr, N, F, stride,
                  BLOCK_N: tl.constexpr, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    fo = tl.arange(0, BLOCK_F)
    fmask = fo < F
    m2 = rmask[:, None] & fmask[None, :]
    rows64 = rows.to(tl.int64)
    start = tl.load(rowptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(rowptr_ptr + rows + 1, mask=rmask, other=0)
    deg = end - start
    hi = tl.load(h_ptr + rows64[:, None] * stride + fo[None, :], mask=m2, other=0.0)
    acc = tl.zeros([BLOCK_N, BLOCK_F], dtype=tl.float32)
    maxdeg = tl.max(deg, axis=0)
    for k in range(0, maxdeg, 1):  # explicit step: Triton 3.5 + Python 3.14 bug (ast.Num)
        valid = (k < deg) & rmask
        j = tl.load(col_ptr + start + k, mask=valid, other=0).to(tl.int64)
        vm = valid[:, None] & fmask[None, :]
        hj = tl.load(h_ptr + j[:, None] * stride + fo[None, :], mask=vm, other=0.0)
        acc += tl.where(vm, tl.abs(hi - hj).to(tl.float32), 0.0)
    inv = 1.0 / tl.maximum(deg, 1).to(tl.float32)
    tl.store(out_ptr + rows64[:, None] * stride + fo[None, :], (acc * inv[:, None]).to(out_ptr.dtype.element_ty),
             mask=m2)


@triton.jit
def _contrast_bwd(h_ptr, g_ptr, rowptr_ptr, col_ptr, gin_ptr, N, F, stride,
                  BLOCK_N: tl.constexpr, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    fo = tl.arange(0, BLOCK_F)
    fmask = fo < F
    m2 = rmask[:, None] & fmask[None, :]
    rows64 = rows.to(tl.int64)
    start = tl.load(rowptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(rowptr_ptr + rows + 1, mask=rmask, other=0)
    deg = end - start
    hk = tl.load(h_ptr + rows64[:, None] * stride + fo[None, :], mask=m2, other=0.0).to(tl.float32)
    gk = tl.load(g_ptr + rows64[:, None] * stride + fo[None, :], mask=m2, other=0.0).to(tl.float32)
    gk = gk / tl.maximum(deg, 1).to(tl.float32)[:, None]
    acc = tl.zeros([BLOCK_N, BLOCK_F], dtype=tl.float32)
    maxdeg = tl.max(deg, axis=0)
    for k in range(0, maxdeg, 1):  # explicit step: Triton 3.5 + Python 3.14 bug (ast.Num)
        valid = (k < deg) & rmask
        j = tl.load(col_ptr + start + k, mask=valid, other=0)
        degj = tl.load(rowptr_ptr + j + 1, mask=valid, other=1) - tl.load(rowptr_ptr + j, mask=valid, other=0)
        j64 = j.to(tl.int64)
        vm = valid[:, None] & fmask[None, :]
        hj = tl.load(h_ptr + j64[:, None] * stride + fo[None, :], mask=vm, other=0.0).to(tl.float32)
        gj = tl.load(g_ptr + j64[:, None] * stride + fo[None, :], mask=vm, other=0.0).to(tl.float32)
        gj = gj / tl.maximum(degj, 1).to(tl.float32)[:, None]
        d = hk - hj
        sgn = tl.where(d > 0, 1.0, tl.where(d < 0, -1.0, 0.0))
        acc += tl.where(vm, sgn * (gk + gj), 0.0)
    tl.store(gin_ptr + rows64[:, None] * stride + fo[None, :], acc.to(gin_ptr.dtype.element_ty), mask=m2)


def build_csr(edge_index, n, check_symmetric=True):
    """edge_index [2, E] (row 0 = source j, row 1 = destination i) -> (rowptr int32 [n+1], col int32 [E])."""
    src, dst = edge_index[0].long(), edge_index[1].long()
    if check_symmetric:
        a = torch.sort(src * n + dst).values
        b = torch.sort(dst * n + src).values
        if not torch.equal(a, b):
            raise ValueError("graph is not symmetric; the fused backward assumes it is")
    order = torch.argsort(dst * n + src)  # sort by destination (then source, for locality)
    col = src[order].to(torch.int32).contiguous()
    counts = torch.bincount(dst, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=torch.int64, device=edge_index.device)
    rowptr[1:] = torch.cumsum(counts, 0)
    assert rowptr[-1] < 2 ** 31
    return rowptr.to(torch.int32).contiguous(), col


def _cfg(F):
    BF = max(16, triton.next_power_of_2(F))
    BN = max(1, min(128, 4096 // BF))  # about 4k elements per program
    return BN, BF


class _ContrastFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, rowptr, col, block_n=None):
        h = h.contiguous()
        N, F = h.shape
        BN, BF = _cfg(F)
        BN = block_n or BN
        out = torch.empty_like(h)
        _contrast_fwd[(triton.cdiv(N, BN),)](h, rowptr, col, out, N, F, h.stride(0), BLOCK_N=BN, BLOCK_F=BF)
        ctx.save_for_backward(h, rowptr, col)
        ctx.bn = BN
        return out

    @staticmethod
    def backward(ctx, g):
        h, rowptr, col = ctx.saved_tensors
        g = g.contiguous()
        N, F = h.shape
        _, BF = _cfg(F)
        gin = torch.empty_like(h)
        _contrast_bwd[(triton.cdiv(N, ctx.bn),)](h, g, rowptr, col, gin, N, F, h.stride(0),
                                                 BLOCK_N=ctx.bn, BLOCK_F=BF)
        return gin, None, None, None


def contrast_fused(h, rowptr, col, block_n=None):
    return _ContrastFn.apply(h, rowptr, col, block_n)


def contrast_reference(h, edge_index):
    """Exactly the operation used in models.ContrastConv (PyTorch Geometric scatter)."""
    from torch_geometric.utils import scatter
    src, dst = edge_index[0], edge_index[1]
    return scatter((h[dst] - h[src]).abs(), dst, dim=0, dim_size=h.size(0), reduce="mean")
