"""Seed check: how much of the spread between trainable GNNs on the synthetic data is training noise?

Reruns 5 trainable models with 3 seeds (model init + batch order; data and folds fixed) on the
hard V and M scenarios, using 01_synthetic_pilot.py unchanged. Seed 0 reproduces its result.
Cached in results/synthetic/cache/seedcheck/; summary in results/synthetic/seedcheck.json.
"""
import importlib.util
import json
import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pilot", os.path.join(HERE, "01_synthetic_pilot.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

OUT = os.path.join(P.OUT, "seedcheck")
os.makedirs(OUT, exist_ok=True)
MODELS = ["GCN", "GCN" + P.SDPOOL, "SAGE", "PNA", "Contrast-GNN"]
SEEDS = [0, 1, 2]
SCEN = ["V", "M"]


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    res = []
    for sc in SCEN:
        seed = {"H": 10, "V": 20, "M": 30}[sc] + 1
        graphs = P.make_dataset(sc, "hard", n_per_class=100, seed=seed)
        y = np.array([int(g.y.item()) for g in graphs])
        folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
        for m in MODELS:
            for s in SEEDS:
                f = os.path.join(OUT, f"{sc}_{m.replace(' ', '_').replace('[', '').replace(']', '').replace('+', 'p')}_s{s}.pkl")
                if os.path.exists(f):
                    oof = pickle.load(open(f, "rb"))
                else:
                    t0 = time.time()
                    oof = P.run_gnn(graphs, y, m, folds, epochs=40, seed=s)
                    pickle.dump(oof, open(f, "wb"))
                    print(f"  {sc} {m:22s} seed {s} {time.time() - t0:6.1f}s", flush=True)
                res.append({"scenario": sc, "model": m, "seed": s, "auc": roc_auc_score(y, oof)})
    json.dump(res, open(os.path.join(os.path.dirname(P.OUT), "seedcheck.json"), "w"), indent=1)
    print(f"\n{'scen':4s} {'model':22s} " + " ".join(f"seed{s:d}" for s in SEEDS) + "   mean    range")
    for sc in SCEN:
        for m in MODELS:
            a = [r["auc"] for r in res if r["scenario"] == sc and r["model"] == m]
            print(f"{sc:4s} {m:22s} " + " ".join(f"{x:5.3f}" for x in a) +
                  f"   {np.mean(a):5.3f}  {max(a) - min(a):5.3f}")


if __name__ == "__main__":
    main()
