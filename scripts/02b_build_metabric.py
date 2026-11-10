"""Build METABRIC IMC cell graphs + patient survival table (runs locally, CPU).

Inputs (data/metabric/MBTMEIMCPublic/, Zenodo 7324285 v2.1.0):
  SingleCells.csv     one row per cell: ImageNumber, ObjectNumber, metabric_id, markers, Location_Center_X/Y
  CellNeighbours.csv  authors' neighbour pairs per image (ImageNumber, from, to = ObjectNumber)
  IMCClinical.csv     yearsToStatus, DeathBreast (disease-specific survival), Grade, ER, PAM50, isValidation
Plus overall survival (OS_MONTHS, OS_STATUS) from the public cBioPortal study brca_metabric (cached JSON).

Outputs (data/metabric/processed/):
  graphs.pt     list of dicts per image: image, patient, x (float16, arcsinh markers), pos, edge_index (int32)
  patients.csv  one row per patient with OS and DSS
  markers.txt
  folds.csv     fixed 5-fold patient split, stratified by DSS event (seed 2026), shared by all models
Rules: drop DNA1, DNA2, Histone H3 (nuclear / housekeeping); drop images with < 100 cells (SpaCEy rule).
"""
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "metabric", "MBTMEIMCPublic")
OUT = os.path.join(ROOT, "data", "metabric", "processed")
os.makedirs(OUT, exist_ok=True)
MIN_CELLS = 100


def cbioportal_os():
    f = os.path.join(OUT, "cbioportal_brca_metabric_patient.json")
    if not os.path.exists(f):
        url = ("https://www.cbioportal.org/api/studies/brca_metabric/clinical-data"
               "?clinicalDataType=PATIENT&projection=SUMMARY")
        with urllib.request.urlopen(url) as r, open(f, "wb") as fh:
            fh.write(r.read())
    d = pd.DataFrame(json.load(open(f)))
    w = d.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    out = pd.DataFrame({"metabric_id": w.index,
                        "os_months": pd.to_numeric(w["OS_MONTHS"], errors="coerce").values,
                        "os_event": w["OS_STATUS"].astype(str).str.startswith("1").astype(float).values,
                        "vital_status": w.get("VITAL_STATUS")})
    out.loc[w["OS_STATUS"].isna().values, "os_event"] = np.nan
    return out


def main():
    sc = pd.read_csv(os.path.join(RAW, "SingleCells.csv"))
    cols = list(sc.columns)
    markers = cols[cols.index("Histone H3"):cols.index("DNA2") + 1]
    markers = [m for m in markers if m not in ("DNA1", "DNA2", "Histone H3")]
    print(len(markers), "markers:", markers)

    nb = pd.read_csv(os.path.join(RAW, "CellNeighbours.csv"), usecols=["ImageNumber", "from", "to"])
    nb_by_img = {k: v for k, v in nb.groupby("ImageNumber")}

    graphs, n_drop, n_badedge = [], 0, 0
    for img, df in sc.groupby("ImageNumber"):
        if len(df) < MIN_CELLS:
            n_drop += 1
            continue
        pid = df.metabric_id.iloc[0]
        assert (df.metabric_id == pid).all()
        obj = df.ObjectNumber.to_numpy()
        idx = pd.Series(np.arange(len(df)), index=obj)
        e = nb_by_img.get(img)
        if e is None:
            ei = np.zeros((2, 0), dtype=np.int32)
        else:
            ok = e["from"].isin(idx.index) & e["to"].isin(idx.index)
            n_badedge += int((~ok).sum())
            src = idx[e["from"][ok]].to_numpy()
            dst = idx[e["to"][ok]].to_numpy()
            ei = np.unique(np.concatenate([np.stack([src, dst]), np.stack([dst, src])], 1), axis=1)
            ei = ei[:, ei[0] != ei[1]].astype(np.int32)
        x = np.arcsinh(df[markers].to_numpy(np.float32))
        pos = df[["Location_Center_X", "Location_Center_Y"]].to_numpy(np.float32)
        graphs.append({"image": int(img), "patient": pid, "x": torch.from_numpy(x).half(),
                       "pos": torch.from_numpy(pos), "edge_index": torch.from_numpy(ei)})

    deg = np.array([g["edge_index"].shape[1] / g["x"].shape[0] for g in graphs])
    print(f"images kept {len(graphs)}, dropped (<{MIN_CELLS} cells) {n_drop}, "
          f"edges with unknown cell ids {n_badedge}; mean degree median {np.median(deg):.2f} "
          f"(min {deg.min():.2f}, max {deg.max():.2f}); images with 0 edges {(deg == 0).sum()}")

    cl = pd.read_csv(os.path.join(RAW, "IMCClinical.csv"))
    cl = cl.rename(columns={"yearsToStatus": "dss_years", "DeathBreast": "dss_event"})
    os_ = cbioportal_os()
    pt = cl.merge(os_, on="metabric_id", how="left")
    have = set(g["patient"] for g in graphs)
    pt = pt[pt.metabric_id.isin(have)].reset_index(drop=True)
    print(f"patients with graphs + clinical: {len(pt)}; OS available {pt.os_months.notna().sum()} "
          f"(events {int(pt.os_event.sum())}); DSS events {int(pt.dss_event.sum())}")
    # sanity: DSS follow-up (years) vs OS follow-up (months) should agree for most patients
    agree = np.isclose(pt.dss_years * 12, pt.os_months, atol=1.5).mean()
    print(f"follow-up agreement DSS years*12 vs OS months (|diff|<=1.5 mo): {agree:.2f}")

    graphs = [g for g in graphs if g["patient"] in set(pt.metabric_id)]
    torch.save(graphs, os.path.join(OUT, "graphs.pt"))
    pt.to_csv(os.path.join(OUT, "patients.csv"), index=False)
    open(os.path.join(OUT, "markers.txt"), "w").write("\n".join(markers))
    # fixed 5-fold patient split, stratified by DSS event, shared by every model so comparisons are paired
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import survival_utils as S
    fold = np.zeros(len(pt), int)
    for k, ii in enumerate(S.strat_folds(pt.dss_event.to_numpy(int), 5, np.random.default_rng(2026))):
        fold[ii] = k
    pd.DataFrame({"metabric_id": pt.metabric_id, "fold": fold}).to_csv(os.path.join(OUT, "folds.csv"), index=False)
    print("saved", len(graphs), "graphs,", len(pt), "patients, folds.csv ->", OUT)


if __name__ == "__main__":
    main()
