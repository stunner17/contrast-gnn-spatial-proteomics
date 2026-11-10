"""Synthetic benchmark. Contrast vs averaging GNNs on simulated tissue graphs.

Three scenarios, each at two difficulty levels; 200 patients (100 per class), ~600 cells each,
10 markers, two cell types, Delaunay graphs:
  H (heterogeneity): label = how spatially mixed the two cell types are. Global cell-type
    proportions (exactly 50/50) and marker distributions are identical between classes, so the
    label lives only in local cell-to-cell differences.
  V (variance): label = within-type marker noise SD (1.0 vs 1.0+delta), random arrangement.
    Global SD sees it; mean abundance does not.
  M (mean, control): label = cell-type proportion (0.5 -/+ d), random arrangement.
    Averaging should be fine here; contrast should not be needed.

Models: no-graph baselines (pseudobulk mean, mean+SD, DeepSets), averaging GNNs (GCN, SAGE,
GAT, GIN, PNA with std aggregator), high-pass baselines (FAGCN, ACM-lite), and contrast
operators (GRAD non-trainable, LC+GCN, trainable Contrast-GNN).

Evaluation: 5-fold stratified CV, out-of-fold AUC; p-value from 2000 label permutations of the
out-of-fold scores (measured null); paired permutation of rank-scores vs the best averaging
model, Holm-corrected. These p-values do not include training variance (pilot).

Out-of-fold scores are cached per (scenario, level, model) in results/synthetic/cache/, so reruns skip
done work. The summary goes to results/synthetic/summary_<levels>.json.
Usage: python scripts/01_synthetic_pilot.py [--quick] [--levels hard]
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Delaunay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (FAConv, GATConv, GCNConv, GINConv, PNAConv, SAGEConv,
                                global_mean_pool)
from torch_geometric.utils import degree, scatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "synthetic", "cache")
os.makedirs(OUT, exist_ok=True)

N_MARKERS = 10
MU_A = np.array([1.5] * 4 + [0.0] * 4 + [0.0, 0.0])
MU_B = np.array([0.0] * 4 + [1.5] * 4 + [0.0, 0.0])

# scenario -> level -> (class0 param, class1 param)
SCENARIOS = {
    "H": {"easy": (0.3, 0.9), "hard": (0.5, 0.7)},      # fraction of cells shuffled (mixing)
    "V": {"easy": (1.0, 1.10), "hard": (1.0, 1.04)},    # within-type noise SD
    "M": {"easy": (0.45, 0.55), "hard": (0.48, 0.52)},  # proportion of type A
}


# ----------------------------------------------------------------------------- simulation
def smooth_field(pos, rng, n_waves=6):
    f = np.zeros(len(pos))
    for _ in range(n_waves):
        w = rng.normal(size=2)
        w = w / np.linalg.norm(w) * rng.uniform(2.0, 4.0) * 2 * np.pi
        f += rng.normal() * np.cos(pos @ w + rng.uniform(0, 2 * np.pi))
    return f


def delaunay_edges(pos, prune_pct=95):
    tri = Delaunay(pos)
    s = tri.simplices
    e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    L = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
    e = e[L <= np.percentile(L, prune_pct)]
    return np.concatenate([e, e[:, ::-1]]).T  # both directions


def simulate_tissue(scenario, param, rng, n_cells=600):
    pos = rng.uniform(0, 1, size=(n_cells, 2))
    sigma, prop_a, mix = 1.0, 0.5, 1.0
    if scenario == "H":
        mix = param
    elif scenario == "V":
        sigma = param
    elif scenario == "M":
        prop_a = param
    # segregated assignment from a smooth field, thresholded at the target proportion
    f = smooth_field(pos, rng)
    is_a = f > np.quantile(f, 1 - prop_a)
    # shuffle types among a random fraction `mix` of cells (keeps proportions exact)
    idx = rng.choice(n_cells, size=int(round(mix * n_cells)), replace=False)
    is_a[idx] = rng.permutation(is_a[idx])
    mu = np.where(is_a[:, None], MU_A, MU_B)
    x = mu + rng.normal(scale=sigma, size=(n_cells, N_MARKERS))
    return pos, x.astype(np.float32), delaunay_edges(pos)


def make_dataset(scenario, level, n_per_class=100, seed=0):
    rng = np.random.default_rng(seed)
    p0, p1 = SCENARIOS[scenario][level]
    graphs = []
    for y, p in [(0, p0), (1, p1)]:
        for _ in range(n_per_class):
            pos, x, ei = simulate_tissue(scenario, p, rng)
            graphs.append(Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(ei).long(),
                               y=torch.tensor([float(y)])))
    return graphs


# ----------------------------------------------------------------------------- operators
def mean_neighbors(x, ei):
    return scatter(x[ei[0]], ei[1], dim=0, dim_size=x.size(0), reduce="mean")


def grad_op(x, ei):
    """GRAD: g_i = mean_j |x_i - x_j|."""
    return scatter((x[ei[1]] - x[ei[0]]).abs(), ei[1], dim=0, dim_size=x.size(0), reduce="mean")


class ContrastConv(nn.Module):
    """Trainable contrast layer: h_i' = MLP([h_i, mean_j |W h_i - W h_j|])."""

    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, h, ei):
        wh = self.W(h)
        c = scatter((wh[ei[1]] - wh[ei[0]]).abs(), ei[1], dim=0, dim_size=h.size(0), reduce="mean")
        return self.mlp(torch.cat([h, c], dim=1))


class ACMLite(nn.Module):
    """Low-pass + high-pass + identity channels (simplified ACM-GNN)."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(3 * d, d)

    def forward(self, h, ei):
        m = mean_neighbors(h, ei)
        return self.lin(torch.cat([h, m, h - m], dim=1))


class Net(nn.Module):
    def __init__(self, kind, d_in=N_MARKERS, d=32, n_layers=2, deg_hist=None, pool="mean"):
        super().__init__()
        self.kind = kind
        self.pool = pool
        d_first = 2 * d_in if kind == "LC+GCN" else d_in
        self.enc = nn.Linear(d_first, d)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            if kind in ("GCN", "LC+GCN"):
                c = GCNConv(d, d)
            elif kind == "SAGE":
                c = SAGEConv(d, d, aggr="mean")
            elif kind == "GAT":
                c = GATConv(d, d // 4, heads=4)
            elif kind == "GIN":
                c = GINConv(nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)))
            elif kind == "PNA":
                c = PNAConv(d, d, aggregators=["mean", "min", "max", "std"],
                            scalers=["identity"], deg=deg_hist)
            elif kind == "FAGCN":
                c = FAConv(d, eps=0.1)
            elif kind == "ACM-lite":
                c = ACMLite(d)
            elif kind == "Contrast-GNN":
                c = ContrastConv(d)
            elif kind == "DeepSets":
                c = None
            else:
                raise ValueError(kind)
            self.convs.append(c)
        d_pool = 2 * d if pool == "meansd" else d
        # BatchNorm on the pooled patient vector: between-patient differences after pooling are
        # tiny vs the overall scale; without it PNA / Contrast-GNN outputs were constant (v1 bug)
        self.pool_norm = nn.BatchNorm1d(d_pool)
        self.head = nn.Sequential(nn.Linear(d_pool, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        if self.kind == "LC+GCN":
            x = torch.cat([x, (x - mean_neighbors(x, ei)).abs()], dim=1)
        h = F.relu(self.enc(x))
        h0 = h
        for c in self.convs:
            if c is None:
                h = h
            elif self.kind == "FAGCN":
                h = F.relu(c(h, h0, ei))
            else:
                h = F.relu(c(h, ei))
        z = global_mean_pool(h, data.batch)
        if self.pool == "meansd":
            sd = (global_mean_pool(h * h, data.batch) - z * z).clamp_min(1e-8).sqrt()
            z = torch.cat([z, sd], dim=1)
        return self.head(self.pool_norm(z)).squeeze(-1)


BASE_GNNS = ["DeepSets", "GCN", "SAGE", "GAT", "GIN", "PNA", "FAGCN", "ACM-lite",
             "LC+GCN", "Contrast-GNN"]
SDPOOL = " [mean+SD pool]"
GNN_MODELS = BASE_GNNS + [m + SDPOOL for m in BASE_GNNS]


def split_model(name):
    return (name[:-len(SDPOOL)], "meansd") if name.endswith(SDPOOL) else (name, "mean")

LR_MODELS = ["Mean (no graph)", "Mean+SD (no graph)", "GRAD (non-trainable)"]
AVERAGING = ["GCN", "SAGE", "GAT", "GIN", "PNA"]  # mean-pool versions = the standard practice
CONTRAST = ["GRAD (non-trainable)", "LC+GCN", "Contrast-GNN",
            "LC+GCN" + SDPOOL, "Contrast-GNN" + SDPOOL]


# ----------------------------------------------------------------------------- features for LR
def patient_features(g, model):
    x = g.x
    if model == "Mean (no graph)":
        return x.mean(0).numpy()
    if model == "Mean+SD (no graph)":
        return torch.cat([x.mean(0), x.std(0)]).numpy()
    if model == "GRAD (non-trainable)":
        gr = grad_op(x, g.edge_index)
        return torch.cat([x.mean(0), x.std(0), gr.mean(0), gr.std(0)]).numpy()
    raise ValueError(model)


# ----------------------------------------------------------------------------- CV runners
def run_lr(graphs, y, model, folds):
    X = np.stack([patient_features(g, model) for g in graphs])
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def run_gnn(graphs, y, model, folds, epochs, seed=0):
    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        torch.manual_seed(seed * 100 + k)
        deg_hist = None
        if model.startswith("PNA"):
            mx = max(int(degree(graphs[i].edge_index[1], graphs[i].num_nodes).max()) for i in tr)
            deg_hist = torch.zeros(mx + 1, dtype=torch.long)
            for i in tr:
                d = degree(graphs[i].edge_index[1], graphs[i].num_nodes, dtype=torch.long)
                deg_hist += torch.bincount(d, minlength=deg_hist.numel())
        base, pool = split_model(model)
        net = Net(base, deg_hist=deg_hist, pool=pool)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        tr_loader = DataLoader([graphs[i] for i in tr], batch_size=16, shuffle=True)
        te_loader = DataLoader([graphs[i] for i in te], batch_size=64, shuffle=False)
        for _ in range(epochs):
            net.train()
            for b in tr_loader:
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(net(b), b.y)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            oof[te] = torch.cat([torch.sigmoid(net(b)) for b in te_loader]).numpy()
    return oof


# ----------------------------------------------------------------------------- statistics
def perm_p_auc(y, s, n=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    a = roc_auc_score(y, s)
    null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(n)])
    return a, (1 + np.sum(null >= a)) / (n + 1)


def paired_perm_p(y, s1, s2, n=2000, rng=None):
    """One-sided: is model 1 better than model 2? Swap rank-scores per patient."""
    rng = rng or np.random.default_rng(1)
    r1 = np.argsort(np.argsort(s1)) / (len(s1) - 1)
    r2 = np.argsort(np.argsort(s2)) / (len(s2) - 1)
    obs = roc_auc_score(y, r1) - roc_auc_score(y, r2)
    cnt = 0
    for _ in range(n):
        sw = rng.random(len(y)) < 0.5
        a, b = np.where(sw, r2, r1), np.where(sw, r1, r2)
        cnt += (roc_auc_score(y, a) - roc_auc_score(y, b)) >= obs
    return obs, (1 + cnt) / (n + 1)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


# ----------------------------------------------------------------------------- diagnostics
def variance_diagnostic(graphs):
    """Per-marker variance across cells: raw, after 1 mean aggregation (with self), after GRAD."""
    raw, avg, grd = [], [], []
    for g in graphs:
        x, ei = g.x, g.edge_index
        m = (scatter(x[ei[0]], ei[1], dim=0, dim_size=x.size(0), reduce="sum") + x) / \
            (degree(ei[1], x.size(0)).unsqueeze(1) + 1)
        raw.append(x.var(0).mean().item())
        avg.append(m.var(0).mean().item())
        grd.append(grad_op(x, ei).var(0).mean().item())
    return float(np.mean(raw)), float(np.mean(avg)), float(np.mean(grd))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--levels", default="hard", help="comma list: easy, hard (reported: hard)")
    args = ap.parse_args()
    n_per_class, epochs = (20, 3) if args.quick else (100, 40)
    tag = "quick" if args.quick else "full"
    sig = hashlib.md5(json.dumps(["v2-poolnorm", SCENARIOS, n_per_class, epochs, LR_MODELS,
                                  MU_A.tolist(), MU_B.tolist()]).encode()).hexdigest()[:8]
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    summary = []
    for sc in SCENARIOS:
        for lvl in [l for l in SCENARIOS[sc] if l in args.levels.split(",")]:
            seed = {"H": 10, "V": 20, "M": 30}[sc] + (lvl == "hard")
            graphs = make_dataset(sc, lvl, n_per_class=n_per_class, seed=seed)
            y = np.array([int(g.y.item()) for g in graphs])
            folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
            vraw, vavg, vgrd = variance_diagnostic(graphs)
            print(f"\n=== scenario {sc} / {lvl}: variance raw {vraw:.3f} -> 1x mean-agg {vavg:.3f} "
                  f"({vavg / vraw:.2f}x); GRAD {vgrd:.3f}", flush=True)
            scores = {}
            for model in LR_MODELS + GNN_MODELS:
                f = os.path.join(OUT, f"{tag}_{sig}_{sc}_{lvl}_{model.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p').replace('[', '').replace(']', '')}.pkl")
                if os.path.exists(f):
                    with open(f, "rb") as fh:
                        oof = pickle.load(fh)
                else:
                    t0 = time.time()
                    oof = run_lr(graphs, y, model, folds) if model in LR_MODELS else \
                        run_gnn(graphs, y, model, folds, epochs)
                    with open(f, "wb") as fh:
                        pickle.dump(oof, fh)
                    print(f"  {model:22s} {time.time() - t0:6.1f}s", flush=True)
                scores[model] = oof
            rows = {}
            for model, s in scores.items():
                a, p = perm_p_auc(y, s)
                rows[model] = {"auc": a, "p_vs_chance": p}
            best_avg = max(AVERAGING, key=lambda m: rows[m]["auc"])
            comp = [(m, *paired_perm_p(y, scores[m], scores[best_avg])) for m in CONTRAST]
            adj = holm([c[2] for c in comp])
            for (m, d, p), pa in zip(comp, adj):
                rows[m]["delta_vs_best_avg"] = d
                rows[m]["p_vs_best_avg_holm"] = pa
            for m, r in rows.items():
                summary.append({"scenario": sc, "level": lvl, "model": m, "best_avg": best_avg,
                                "var_raw": vraw, "var_meanagg": vavg, **r})
            print(f"  {'model':22s} {'AUC':>6s} {'p(chance)':>10s} {'dAUC vs ' + best_avg:>14s} {'p Holm':>8s}")
            for m in LR_MODELS + GNN_MODELS:
                r = rows[m]
                extra = (f"{r['delta_vs_best_avg']:+14.3f} {r['p_vs_best_avg_holm']:8.4f}"
                         if "delta_vs_best_avg" in r else "")
                print(f"  {m:22s} {r['auc']:6.3f} {r['p_vs_chance']:10.4f} {extra}", flush=True)
    with open(os.path.join(os.path.dirname(OUT), f"summary_{args.levels.replace(',', '-')}_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nsaved summary")


if __name__ == "__main__":
    main()
