"""Analyse the GPU GNN run (results/02_gnn/preds, 1000 tasks from gnn/train_gnn.py) on the pre-specified panel.

Per model and endpoint: test-fold risks are rank-normalised within each fold, then pooled into one
out-of-fold (OOF) score per patient; the 5 seeds are averaged into a seed ensemble (primary score).
Ridge-Cox baselines (Mean, Mean+SD, GRAD, Clinical) are refit on the SAME folds (folds.csv) so every
comparison is paired on identical splits.

Stats (all measured, no assumed 0.5):
  null      : 5000 permutations of the OOF score against (time, event). Valid because scores are OOF.
  paired    : two-sided paired permutation (swap the two models' scores per patient), 5000 draws.
  Holm      : across all tests of a family, across tasks.
Families:
  PRIMARY   (pre-specified) graph model vs matched no-graph model (DeepSets, same pooling),
            9 graph architectures x 2 poolings x 3 DSS tasks (all, ER+, ER-) = 54 tests.
  SECONDARY contrast vs averaging (Contrast-GNN vs GCN/GAT/GIN/SAGE, both poolings), SD vs mean
            pooling per architecture, best GNNs vs ridge baselines, OS endpoint. Each Holm within itself.
"""
import importlib.util
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import survival_utils as S  # noqa: E402

PRED = os.path.join(ROOT, "results", "02_gnn", "preds")
PROC = os.path.join(ROOT, "data", "metabric", "processed")
OUT = os.path.join(ROOT, "results", "metabric")
os.makedirs(OUT, exist_ok=True)
NPERM = 5000
ARCHS = ["DeepSets", "GCN", "SAGE", "GAT", "GIN", "PNA", "FAGCN", "ACM-lite", "LC+GCN", "Contrast-GNN"]
GRAPH = ARCHS[1:]
POOLS = ["", "+SDpool"]
LOG = open(os.path.join(OUT, "analysis_log.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


# ---------- fast C-index on fixed comparable pairs (vectorised over many score vectors) ----------
class Pairs:
    def __init__(self, t, e):
        i, j = np.where((e[:, None] == 1) & (t[:, None] < t[None, :]))
        self.i, self.j = i, j

    def c(self, R):
        """R: (n,) or (B, n) scores -> C-index (Harrell)."""
        R = np.atleast_2d(R)
        a, b = R[:, self.i], R[:, self.j]
        return ((a > b).sum(1) + 0.5 * (a == b).sum(1)) / len(self.i)


def perm_p(P, r, rng):
    obs = P.c(r)[0]
    null = np.concatenate([P.c(np.stack([rng.permutation(r) for _ in range(100)])) for _ in range(NPERM // 100)])
    return obs, (1 + (null >= obs).sum()) / (len(null) + 1), null.mean(), np.quantile(null, .95)


def paired(P, r1, r2, rng):
    obs = P.c(r1)[0] - P.c(r2)[0]
    cnt = 0
    for _ in range(NPERM // 100):
        s = rng.random((100, len(r1))) < 0.5
        d = P.c(np.where(s, r2, r1)) - P.c(np.where(s, r1, r2))
        cnt += (np.abs(d) >= abs(obs) - 1e-12).sum()
    return obs, (1 + cnt) / (NPERM + 1)


def ranknorm(x):
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)


# ---------- load GNN predictions ----------
def load_gnn(ep, ids):
    pos = {p: k for k, p in enumerate(ids)}
    out, meta = {}, []
    for a in ARCHS:
        for pl in POOLS:
            m = a + pl
            seeds = []
            for s in range(5):
                r = np.full(len(ids), np.nan)
                for f in range(5):
                    d = np.load(os.path.join(PRED, f"{ep}_{m.replace('+', 'p')}_f{f}_s{s}.npz"), allow_pickle=True)
                    r[[pos[p] for p in d["patient"]]] = ranknorm(d["risk"].astype(float))
                    meta.append({"endpoint": ep, "model": m, "fold": f, "seed": s, "best_epoch": int(d["best_epoch"]),
                                 "epochs_run": int(d["epochs_run"]), "sec_train": float(d["sec_train"]),
                                 "val_c": float(d["val_c"]), "test_c": float(d["test_c"])})
                assert not np.isnan(r).any(), (ep, m, s)
                seeds.append(r)
            out[m] = np.stack(seeds)  # (5 seeds, n)
    return out, pd.DataFrame(meta)


# ---------- ridge baselines on the same folds ----------
def ridge_baselines(pts, fold, t, e, ok):
    cache = os.path.join(OUT, "patient_features.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        sets = {k: z[k] for k in z.files}
    else:
        spec = importlib.util.spec_from_file_location("b02c", os.path.join(ROOT, "scripts", "02c_baselines_metabric.py"))
        b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b)
        import torch
        graphs = torch.load(os.path.join(PROC, "graphs.pt"))
        sets = b.patient_table(graphs, pts)
        np.savez(cache, **sets)
    keep = ["Mean (no graph)", "Mean+SD (no graph)", "GRAD (graph contrast)", "Clinical (reference)"]
    res = {}
    rng = np.random.default_rng(7)
    f, tt, ee = fold[ok], t[ok], e[ok]
    for name in keep:
        X = sets[name][ok]
        r = np.zeros(len(tt))
        for k in range(5):
            te, tr = np.where(f == k)[0], np.where(f != k)[0]
            best, bc = 10, -1
            inner = S.strat_folds(ee[tr], 3, rng)
            for lam in (0.1, 1, 10, 100):
                cs = []
                for ite in inner:
                    itr = np.setdiff1d(np.arange(len(tr)), ite)
                    mdl = S.RidgeCox(lam).fit(X[tr][itr], tt[tr][itr], ee[tr][itr])
                    cs.append(S.cindex(tt[tr][ite], ee[tr][ite], mdl.risk(X[tr][ite])))
                if np.mean(cs) > bc:
                    best, bc = lam, np.mean(cs)
            r[te] = ranknorm(S.RidgeCox(best).fit(X[tr], tt[tr], ee[tr]).risk(X[te]))
        res["Ridge: " + name] = r
    return res


def main():
    pts = pd.read_csv(os.path.join(PROC, "patients.csv"))
    folds = pd.read_csv(os.path.join(PROC, "folds.csv")).set_index("metabric_id").loc[pts.metabric_id, "fold"].to_numpy()
    assert len(glob(os.path.join(PRED, "*.npz"))) == 1000
    er = pts.ERStatus.to_numpy()
    eps = {"DSS": (pts.dss_years.to_numpy(float), pts.dss_event.to_numpy(int), np.ones(len(pts), bool)),
           "OS": (pts.os_months.to_numpy(float), pts.os_event.fillna(0).to_numpy(int), pts.os_months.notna().to_numpy())}
    rng = np.random.default_rng(2026)
    rows, comps, metas = [], [], []
    scores_all = {}
    for ep, (t, e, ok) in eps.items():
        ids = pts.metabric_id[ok].tolist()
        gnn, meta = load_gnn(ep, ids)
        metas.append(meta)
        ridge = ridge_baselines(pts, folds, t, e, ok)
        ens = {m: v.mean(0) for m, v in gnn.items()}
        scores = {**ens, **ridge}
        scores_all[ep] = scores
        tasks = {"all": np.ones(ok.sum(), bool)} if ep == "OS" else \
            {"all": np.ones(ok.sum(), bool), "ER+": er[ok] == "pos", "ER-": er[ok] == "neg"}
        for task, mask in tasks.items():
            tt, ee = t[ok][mask], e[ok][mask]
            P = Pairs(tt, ee)
            say(f"\n=== {ep} / {task}: n={mask.sum()}, events={ee.sum()}, comparable pairs={len(P.i)}")
            for m, r in scores.items():
                c, p, nm, n95 = perm_p(P, r[mask], rng)
                seed_c = P.c(gnn[m][:, mask]) if m in gnn else np.array([c])
                rows.append({"endpoint": ep, "task": task, "model": m, "C_ensemble": c, "C_seed_mean": seed_c.mean(),
                             "C_seed_sd": seed_c.std(), "null_mean": nm, "null_95": n95, "p_perm": p})
                say(f"  {m:34s} C={c:.3f}  (single seeds {seed_c.mean():.3f} +- {seed_c.std():.3f})  "
                    f"null {nm:.3f}/95% {n95:.3f}  p={p:.4f}")

            def add(fam, a, b):
                d, p = paired(P, scores[a][mask], scores[b][mask], rng)
                comps.append({"family": fam, "endpoint": ep, "task": task, "A": a, "B": b, "dC": d, "p": p})

            fam1 = "PRIMARY graph vs no-graph" if ep == "DSS" else "OS graph vs no-graph"
            for a in GRAPH:
                for pl in POOLS:
                    add(fam1, a + pl, "DeepSets" + pl)
            if ep == "DSS":
                for pl in POOLS:
                    for b in ["GCN", "GAT", "GIN", "SAGE"]:
                        add("contrast vs averaging", "Contrast-GNN" + pl, b + pl)
                for a in ARCHS:
                    add("SDpool vs mean pool", a + "+SDpool", a)
                for a in ["Contrast-GNN", "Contrast-GNN+SDpool", "GCN", "DeepSets+SDpool"]:
                    for b in ["Ridge: Mean+SD (no graph)", "Ridge: GRAD (graph contrast)"]:
                        add("GNN vs ridge baseline", a, b)
    C = pd.DataFrame(comps)
    C["p_holm"] = np.nan
    for fam, g in C.groupby("family"):
        C.loc[g.index, "p_holm"] = S.holm(g.p.to_numpy())
    R = pd.DataFrame(rows)
    M = pd.concat(metas)
    R.to_csv(os.path.join(OUT, "cindex_table.csv"), index=False)
    C.to_csv(os.path.join(OUT, "comparisons.csv"), index=False)
    M.to_csv(os.path.join(OUT, "training_meta.csv"), index=False)
    for ep, sc in scores_all.items():
        np.savez(os.path.join(OUT, f"oof_scores_{ep}.npz"), **{k.replace(":", "").replace(" ", "_"): v for k, v in sc.items()})
    say("\n\n######## COMPARISONS (two-sided paired permutation, Holm within family) ########")
    for fam, g in C.groupby("family", sort=False):
        say(f"\n--- {fam} ({len(g)} tests)")
        for _, r in g.iterrows():
            flag = "  *" if r.p_holm < 0.05 else ""
            say(f"  {r.endpoint}/{r.task:4s} {r.A:24s} vs {r.B:30s} dC={r.dC:+.3f}  p={r.p:.4f}  Holm={r.p_holm:.4f}{flag}")
    say("\n--- training: median best epoch / median epochs run / median train s, per model (DSS)")
    say(M[M.endpoint == "DSS"].groupby("model")[["best_epoch", "epochs_run", "sec_train"]].median().round(1).to_string())
    json.dump({"n_perm": NPERM}, open(os.path.join(OUT, "settings.json"), "w"))
    say("\nsaved to", OUT)


if __name__ == "__main__":
    main()
