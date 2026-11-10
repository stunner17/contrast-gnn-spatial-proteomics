"""Correctness and speed of the fused Triton contrast kernel vs PyTorch Geometric.

1. Correctness (gate: the script exits with an error before any timing if this fails)
   random symmetric graphs with isolated nodes, F in {7, 32, 100}; then the full METABRIC graph, F = 32.
   Forward and backward compared with the PyG reference (autograd), max relative error < 1e-4.
2. Operator benchmark on the METABRIC graph (1.1 M cells, 6.4 M edges) and a 4x copy of it
   (4.4 M cells, 25.6 M edges), F in {16, 32, 64, 128}, fp32:
   PyG scatter  vs  torch.compile(PyG expression)  vs  fused Triton.
   Median time (triton.testing.do_bench) for forward and forward+backward, plus peak extra GPU memory.
3. End to end: one full-batch training step (forward + Cox loss + backward + Adam) of Contrast-GNN+SDpool
   on all METABRIC images, with the PyG contrast op vs the fused op.
Writes results/kernel/<gpu>_<jobid>.json and prints a table.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from contrast_triton import build_csr, contrast_fused, contrast_reference  # noqa: E402


def random_symmetric_graph(n, avg_deg, n_isolated, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = n * avg_deg // 2
    a = torch.randint(0, n - n_isolated, (m,), generator=g)
    b = torch.randint(0, n - n_isolated, (m,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    key = torch.unique(torch.minimum(a, b) * n + torch.maximum(a, b))
    a, b = key // n, key % n
    ei = torch.stack([torch.cat([a, b]), torch.cat([b, a])])
    return ei.to(dev)


def rel_err(x, y):
    return ((x - y).abs().max() / y.abs().max().clamp_min(1e-12)).item()


def check(h, ei, tag, tol=1e-4):
    n = h.shape[0]
    rowptr, col = build_csr(ei, n)
    g = torch.randn_like(h)
    h1 = h.clone().requires_grad_(True)
    ref = contrast_reference(h1, ei)
    (ref * g).sum().backward()
    h2 = h.clone().requires_grad_(True)
    out = contrast_fused(h2, rowptr, col)
    (out * g).sum().backward()
    ef, eb = rel_err(out.detach(), ref.detach()), rel_err(h2.grad, h1.grad)
    ok = ef < tol and eb < tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {tag:40s} fwd rel err {ef:.2e}  bwd rel err {eb:.2e}", flush=True)
    return ok


def correctness(dev, metabric=None):
    print("== correctness", flush=True)
    ok = True
    for n, F in [(3000, 7), (3000, 32), (2000, 100)]:
        ei = random_symmetric_graph(n, 8, n_isolated=50, dev=dev)
        ok &= check(torch.randn(n, F, device=dev), ei, f"random n={n} E={ei.shape[1]} F={F}")
    if metabric is not None:
        x, ei = metabric
        gen = torch.Generator(device=dev).manual_seed(1)
        W = torch.randn(x.shape[1], 32, device=dev, generator=gen) / x.shape[1] ** 0.5
        ok &= check(x @ W, ei, f"METABRIC n={x.shape[0]} E={ei.shape[1]} F=32")
    return ok


def load_metabric(data_dir, dev):
    graphs = torch.load(os.path.join(data_dir, "graphs.pt"))
    xs, eis, off = [], [], 0
    for g in graphs:
        xs.append(g["x"].float())
        eis.append(g["edge_index"].long() + off)
        off += g["x"].shape[0]
    return torch.cat(xs).to(dev), torch.cat(eis, 1).to(dev)


def replicate(x, ei, k):
    n = x.shape[0]
    return x.repeat(k, 1), torch.cat([ei + i * n for i in range(k)], 1)


def peak_extra(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def bench_ops(x, ei, Fs, tag):
    from triton.testing import do_bench
    n = x.shape[0]
    rowptr, col = build_csr(ei, n)
    compiled = None
    try:
        compiled = torch.compile(contrast_reference, dynamic=False)
    except Exception as e:  # noqa: BLE001
        print("  torch.compile unavailable:", repr(e)[:200])
    rows = []
    for F in Fs:
        gen = torch.Generator(device=x.device).manual_seed(F)
        W = torch.randn(x.shape[1], F, device=x.device, generator=gen) / x.shape[1] ** 0.5
        h = (x @ W).contiguous().requires_grad_(True)
        g = torch.randn_like(h)
        impls = {"PyG scatter": lambda: contrast_reference(h, ei),
                 "fused Triton": lambda: contrast_fused(h, rowptr, col)}
        if compiled is not None:
            impls["torch.compile"] = lambda: compiled(h, ei)
        for name, f in impls.items():
            try:
                f().backward(g)  # warm-up / compile
                h.grad = None
                t_f = do_bench(lambda: f(), warmup=10, rep=200)
                t_fb = do_bench(lambda: f().backward(g), warmup=10, rep=200, grad_to_none=[h])
                mem_f = peak_extra(lambda: f())
                mem_fb = peak_extra(lambda: f().backward(g))
                h.grad = None
            except Exception as e:  # noqa: BLE001
                print(f"  {name} failed at F={F}: {repr(e)[:200]}", flush=True)
                continue
            r = {"graph": tag, "N": n, "E": ei.shape[1], "F": F, "impl": name, "fwd_ms": t_f, "fwdbwd_ms": t_fb,
                 "fwd_mem_GB": mem_f, "fwdbwd_mem_GB": mem_fb}
            rows.append(r)
            print(f"  {tag:10s} F={F:3d} {name:14s} fwd {t_f:8.3f} ms  fwd+bwd {t_fb:8.3f} ms  "
                  f"mem fwd {mem_f:6.2f} GB  fwd+bwd {mem_fb:6.2f} GB", flush=True)
    return rows


def end_to_end(data_dir, dev, steps=30):
    """Training-step time of Contrast-GNN+SDpool, PyG op vs fused op (identical model and data)."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gnn"))  # models.py, train_gnn.py
    import models
    from train_gnn import load
    D = load(data_dir, "DSS", dev)
    pts = D["pts"]
    x = D["x"]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    ip = torch.as_tensor(D["img_patient"], device=dev)
    t = torch.as_tensor(pts.t.to_numpy(), dtype=torch.float32, device=dev)[ip]
    e = torch.as_tensor(pts.e.to_numpy(), dtype=torch.float32, device=dev)[ip]
    n_img = len(D["img_patient"])
    rowptr, col = build_csr(D["ei"], x.shape[0])
    orig = models.ContrastConv.forward

    def fused_forward(self, h, ei):
        wh = self.W(h)
        c = contrast_fused(wh, rowptr, col)
        return self.mlp(torch.cat([h, c], dim=1))

    out = {}
    for name, fwd in [("PyG scatter", orig), ("fused Triton", fused_forward)]:
        models.ContrastConv.forward = fwd
        torch.manual_seed(0)
        net = models.SurvNet("Contrast-GNN+SDpool", x.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)

        def step():
            opt.zero_grad(set_to_none=True)
            loss = models.cox_loss(net(x, D["ei"], D["gid"], n_img), t, e)
            loss.backward()
            opt.step()
            return loss

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        losses = [step().item() for _ in range(steps)]
        torch.cuda.synchronize()
        ms = (time.time() - t0) / steps * 1000
        out[name] = {"ms_per_step": ms, "peak_GB": torch.cuda.max_memory_allocated() / 2 ** 30,
                     "loss_first": losses[0], "loss_last": losses[-1]}
        print(f"  end-to-end {name:14s} {ms:8.2f} ms/step  peak {out[name]['peak_GB']:.2f} GB  "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    models.ContrastConv.forward = orig
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metabric/processed")
    ap.add_argument("--out", default="results/kernel")
    ap.add_argument("--local-test", action="store_true", help="CPU interpreter: correctness on small graphs only")
    a = ap.parse_args()
    if a.local_test:
        sys.exit(0 if correctness("cpu") else 1)
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name()
    import triton
    print(f"GPU {gpu} | torch {torch.__version__} | triton {triton.__version__}", flush=True)
    x, ei = load_metabric(a.data, dev)
    if not correctness(dev, (x, ei)):
        sys.exit("correctness FAILED: no benchmark run")
    res = {"gpu": gpu, "torch": torch.__version__, "triton": triton.__version__, "ops": []}
    print("== operator benchmark", flush=True)
    res["ops"] += bench_ops(x, ei, [16, 32, 64, 128], "METABRIC")
    x4, ei4 = replicate(x, ei, 4)
    res["ops"] += bench_ops(x4, ei4, [16, 32, 64, 128], "METABRICx4")
    del x4, ei4
    torch.cuda.empty_cache()
    print("== end to end (Contrast-GNN+SDpool training step, all DSS images)", flush=True)
    res["end_to_end"] = end_to_end(a.data, dev)
    os.makedirs(a.out, exist_ok=True)
    short = "".join(c for c in gpu.replace("NVIDIA", "") if c.isalnum())[:24]
    fn = os.path.join(a.out, f"{short}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    json.dump(res, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
