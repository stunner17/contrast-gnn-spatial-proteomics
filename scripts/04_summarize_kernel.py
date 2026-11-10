"""Summarise the kernel benchmark JSONs (one per GPU) into two tables:
results/kernel/kernel_summary.csv  speedup and memory, fused Triton vs PyTorch Geometric, per GPU, graph and F
results/kernel/end_to_end.csv      one full Contrast-GNN+SDpool training step, PyG op vs fused op
Usage: python scripts/04_summarize_kernel.py
"""
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "kernel")
rows, e2e = [], []
for fn in sorted(glob.glob(os.path.join(R, "*.json"))):
    d = json.load(open(fn))
    gpu = "RTX PRO 6000 Blackwell" if "Blackwell" in d["gpu"] else "H200 NVL"
    rows += [{"gpu": gpu, **r} for r in d["ops"]]
    e2e += [{"gpu": gpu, "impl": k, **v} for k, v in d["end_to_end"].items()]
p = pd.DataFrame(rows).pivot_table(index=["gpu", "graph", "F"], columns="impl",
                                   values=["fwd_ms", "fwdbwd_ms", "fwdbwd_mem_GB"])
s = pd.DataFrame({
    "PyG_fwdbwd_ms": p[("fwdbwd_ms", "PyG scatter")], "Triton_fwdbwd_ms": p[("fwdbwd_ms", "fused Triton")],
    "speedup_fwd": p[("fwd_ms", "PyG scatter")] / p[("fwd_ms", "fused Triton")],
    "speedup_fwdbwd": p[("fwdbwd_ms", "PyG scatter")] / p[("fwdbwd_ms", "fused Triton")],
    "PyG_mem_GB": p[("fwdbwd_mem_GB", "PyG scatter")], "Triton_mem_GB": p[("fwdbwd_mem_GB", "fused Triton")],
}).reset_index()
s["mem_ratio"] = s.PyG_mem_GB / s.Triton_mem_GB
s.round(3).to_csv(os.path.join(R, "kernel_summary.csv"), index=False)
E = pd.DataFrame(e2e)
E.round(3).to_csv(os.path.join(R, "end_to_end.csv"), index=False)
pd.set_option("display.width", 200)
print(s.round(2).to_string(index=False))
print(E.round(3).to_string(index=False))
