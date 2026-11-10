"""Non-GNN survival baselines on METABRIC IMC (local CPU).

Patient features (image features averaged over a patient's images, weighted by cell count):
  Clinical (reference only)  grade, lymph-node ordinal, size ordinal, ER status
  Cell-type proportions      authors' cellPhenotype labels (SpaCEy's best baseline type)
  Mean (no graph)            per-marker mean
  Mean+SD (no graph)         per-marker mean + SD (cell-to-cell variability)
  GRAD (graph contrast)      Mean+SD + per-marker mean and SD of g_i = mean_j |x_i - x_j| over graph neighbours
Model: ridge Cox, penalty chosen by inner 3-fold CV; outer 5-fold CV stratified by event, 10 repeats.
Endpoints: DSS (primary, IMCClinical, n=689) and OS (secondary, cBioPortal, n=595).
Stats: null = 200 permutations of (time, event) with the full CV rerun; paired permutation tests on
repeat-averaged out-of-fold risk ranks for planned comparisons, Holm-corrected within endpoint.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import scatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survival_utils as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "metabric", "processed")
RAW = os.path.join(ROOT, "data", "metabric", "MBTMEIMCPublic")
OUT = os.path.join(ROOT, "results", "metabric")
os.makedirs(OUT, exist_ok=True)
LN_MAP = {"0": 0, "1": 1, "2-3": 2, "4-7": 3, "7+": 4}
SIZE_MAP = {"1-5mm": 0, "5-10mm": 1, "10-20mm": 2, "20-50mm": 3, "50+mm": 4}


def image_features(g):
    x = g["x"].float()
    ei = g["edge_index"].long()
    gr = scatter((x[ei[1]] - x[ei[0]]).abs(), ei[1], dim=0, dim_size=x.size(0), reduce="mean")
    return {"mean": x.mean(0).numpy(), "sd": x.std(0).numpy(),
            "gmean": gr.mean(0).numpy(), "gsd": gr.std(0).numpy(), "n": x.size(0)}


def patient_table(graphs, pts):
    feats = {}
    for g in graphs:
        feats.setdefault(g["patient"], []).append(image_features(g))
    rows = {}
    for pid, fl in feats.items():
        w = np.array([f["n"] for f in fl], float)
        w /= w.sum()
        rows[pid] = {k: sum(wi * f[k] for wi, f in zip(w, fl)) for k in ("mean", "sd", "gmean", "gsd")}
    sets = {}
    ids = pts.metabric_id.tolist()
    m = np.stack([rows[i]["mean"] for i in ids])
    sd = np.stack([rows[i]["sd"] for i in ids])
    gm = np.stack([rows[i]["gmean"] for i in ids])
    gs = np.stack([rows[i]["gsd"] for i in ids])
    sets["Mean (no graph)"] = m
    sets["Mean+SD (no graph)"] = np.hstack([m, sd])
    sets["GRAD (graph contrast)"] = np.hstack([m, sd, gm, gs])
    # cell-type proportions
    sc = pd.read_csv(os.path.join(RAW, "SingleCells.csv"), usecols=["ImageNumber", "metabric_id", "cellPhenotype"])
    keep = set(g["image"] for g in graphs)
    sc = sc[sc.ImageNumber.isin(keep)]
    prop = pd.crosstab(sc.metabric_id, sc.cellPhenotype, normalize="index").reindex(ids).fillna(0)
    sets["Cell-type proportions"] = prop.to_numpy()
    # clinical reference
    c = pts.copy()
    ln = c.LymphNodesOrdinal.map(LN_MAP)
    sz = c.sizeOrdinal.map(SIZE_MAP)
    er = c.ERStatus.map({"pos": 1.0, "neg": 0.0})
    clin = pd.DataFrame({"grade": c.Grade, "ln": ln, "size": sz, "er": er})
    clin = clin.fillna(clin.median())
    sets["Clinical (reference)"] = clin.to_numpy(float)
    return sets


def main():
    graphs = torch.load(os.path.join(PROC, "graphs.pt"))
    pts = pd.read_csv(os.path.join(PROC, "patients.csv"))
    print(f"{len(graphs)} graphs, {len(pts)} patients")
    bad = (set(pts.LymphNodesOrdinal.dropna()) - set(LN_MAP)) | (set(pts.sizeOrdinal.dropna()) - set(SIZE_MAP))
    assert not bad, f"unmapped clinical categories: {bad}"
    sets = patient_table(graphs, pts)
    endpoints = {"DSS": (pts.dss_years.to_numpy(), pts.dss_event.to_numpy(int), np.ones(len(pts), bool)),
                 "OS": (pts.os_months.to_numpy(), pts.os_event.fillna(0).to_numpy(int), pts.os_months.notna().to_numpy())}
    planned = [("Mean+SD (no graph)", "Mean (no graph)"),        # does variability add to mean?
               ("GRAD (graph contrast)", "Mean+SD (no graph)"),  # does graph contrast add to variability?
               ("Mean+SD (no graph)", "Cell-type proportions")]
    results = []
    for ep, (t, e, ok) in endpoints.items():
        t, e = t[ok], e[ok]
        print(f"\n=== {ep}: n={ok.sum()}, events={e.sum()}")
        oofs = {}
        for name, X in sets.items():
            X = X[ok]
            cis, oof = S.cv_cox(X, t, e, k=5, repeats=10)
            null = S.perm_null_cv(X, t, e, n_perm=200)
            p = (1 + np.sum(null >= cis.mean())) / (len(null) + 1)
            oofs[name] = oof
            r = {"endpoint": ep, "features": name, "n_features": X.shape[1], "cindex_mean": cis.mean(),
                 "cindex_sd": cis.std(), "null_mean": null.mean(), "null_95": np.quantile(null, .95), "p_perm": p}
            results.append(r)
            print(f"  {name:24s} d={X.shape[1]:3d}  C={cis.mean():.3f} +- {cis.std():.3f}   "
                  f"null {null.mean():.3f} (95% {np.quantile(null, .95):.3f})  p={p:.4f}", flush=True)
        comps = [(a, b, *S.paired_perm_cindex(t, e, oofs[a], oofs[b])) for a, b in planned]
        adj = S.holm([c[3] for c in comps])
        for (a, b, d, p), pa in zip(comps, adj):
            print(f"  {a} vs {b}: dC={d:+.3f}  p={p:.4f}  Holm={pa:.4f}")
            results.append({"endpoint": ep, "comparison": f"{a} vs {b}", "delta_c": d, "p": p, "p_holm": pa})
    json.dump(results, open(os.path.join(OUT, "baselines.json"), "w"), indent=1, default=float)
    print("\nsaved", os.path.join(OUT, "baselines.json"))


if __name__ == "__main__":
    main()
