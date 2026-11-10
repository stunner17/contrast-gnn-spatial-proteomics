"""Does tissue add anything beyond clinical variables? (METABRIC, local CPU)

Same protocol as 02c (ridge Cox, inner-CV penalty, 10 x 5-fold CV, 200-permutation null).
Planned comparisons vs Clinical alone, Holm-corrected within endpoint.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import survival_utils as S  # noqa: E402

spec = importlib.util.spec_from_file_location("b", os.path.join(HERE, "02c_baselines_metabric.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
OUT = os.path.join(B.ROOT, "results", "metabric")


def main():
    graphs = torch.load(os.path.join(B.PROC, "graphs.pt"))
    pts = pd.read_csv(os.path.join(B.PROC, "patients.csv"))
    base = B.patient_table(graphs, pts)
    clin = base["Clinical (reference)"]
    sets = {"Clinical": clin,
            "Clinical + Mean": np.hstack([clin, base["Mean (no graph)"]]),
            "Clinical + Mean+SD": np.hstack([clin, base["Mean+SD (no graph)"]]),
            "Clinical + GRAD": np.hstack([clin, base["GRAD (graph contrast)"]])}
    endpoints = {"DSS": (pts.dss_years.to_numpy(), pts.dss_event.to_numpy(int), np.ones(len(pts), bool)),
                 "OS": (pts.os_months.to_numpy(), pts.os_event.fillna(0).to_numpy(int), pts.os_months.notna().to_numpy())}
    res = []
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
            res.append({"endpoint": ep, "features": name, "cindex_mean": cis.mean(), "cindex_sd": cis.std(), "p_perm": p})
            print(f"  {name:22s} d={X.shape[1]:3d}  C={cis.mean():.3f} +- {cis.std():.3f}  p={p:.4f}", flush=True)
        comps = [(a, *S.paired_perm_cindex(t, e, oofs[a], oofs["Clinical"])) for a in list(sets)[1:]]
        adj = S.holm([c[2] for c in comps])
        for (a, d, p), pa in zip(comps, adj):
            print(f"  {a} vs Clinical: dC={d:+.3f}  p={p:.4f}  Holm={pa:.4f}")
            res.append({"endpoint": ep, "comparison": f"{a} vs Clinical", "delta_c": d, "p": p, "p_holm": pa})
    json.dump(res, open(os.path.join(OUT, "clinical_plus_tissue.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
