"""All figures in the README, from the saved results (no training needed).

fig1  synthetic benchmark: AUC of every model in the three simulated scenarios
fig2  METABRIC: C-index of every model (DSS, all patients / ER+ / ER-), with the permutation null
fig3  METABRIC: each GNN minus its edge-removed twin (DeepSets), with patient-bootstrap 95% CIs
fig4  fused Triton kernel: speedup and memory vs PyTorch Geometric, RTX PRO 6000 Blackwell vs H200

fig3 needs data/metabric/processed/patients.csv (survival times; see scripts/02b_build_metabric.py).
Usage: python scripts/05_figures.py
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})

# Okabe-Ito colours by model family
FAM = {"No graph": "#7F7F7F", "Averaging GNN": "#0072B2", "High-pass GNN": "#CC79A7",
       "Contrast GNN": "#D55E00", "Ridge Cox": "#000000"}
ARCH_FAM = {"Mean (no graph)": "No graph", "Mean+SD (no graph)": "No graph", "DeepSets": "No graph",
            "GCN": "Averaging GNN", "SAGE": "Averaging GNN", "GAT": "Averaging GNN", "GIN": "Averaging GNN",
            "PNA": "Averaging GNN", "FAGCN": "High-pass GNN", "ACM-lite": "High-pass GNN",
            "GRAD (non-trainable)": "Contrast GNN", "LC+GCN": "Contrast GNN", "Contrast-GNN": "Contrast GNN"}
POOL_MARK = {"mean": ("o", "mean pooling"), "sd": ("^", "mean + SD pooling")}


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def family_rows(ax, archs, fam_of):
    """Label y rows and shade alternate families."""
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs)
    for lab, a in zip(ax.get_yticklabels(), archs):
        lab.set_color(FAM[fam_of[a]])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.set_axisbelow(True)


def legend_handles(families):
    h = [Line2D([], [], ls="", marker="s", color=FAM[f], label=f) for f in families]
    h += [Line2D([], [], ls="", marker=m, color="#444444", mfc="none" if k == "mean" else "#444444", label=l)
          for k, (m, l) in POOL_MARK.items()]
    return h


# ------------------------------------------------------------------ fig1 synthetic
def fig1():
    d = pd.DataFrame(json.load(open(os.path.join(RES, "synthetic", "summary_hard_full.json"))))
    d["sd"] = d.model.str.contains(r"\[mean\+SD pool\]", regex=True)
    d["arch"] = d.model.str.replace(" [mean+SD pool]", "", regex=False)
    archs = list(ARCH_FAM)
    titles = {"H": "H: label = spatial mixing\n(same composition, same marker levels)",
              "V": "V: label = cell-to-cell variability\n(random arrangement)",
              "M": "M: label = composition\n(control)"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    for ax, sc in zip(axes, ["H", "V", "M"]):
        q = d[d.scenario == sc]
        for i, a in enumerate(archs):
            c = FAM[ARCH_FAM[a]]
            for sd, off in [(False, -0.15), (True, 0.15)]:
                r = q[(q.arch == a) & (q.sd == sd)]
                if len(r):
                    m = POOL_MARK["sd" if sd else "mean"][0]
                    ax.plot(r.auc.iloc[0], i + off, m, color=c, mfc=c if sd else "white", ms=4.5, mew=1)
        ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
        ax.set_xlim(0.35, 1.02)
        ax.set_xlabel("AUC (5-fold CV)")
        ax.set_title(titles[sc], fontsize=8)
        family_rows(ax, archs, ARCH_FAM)
    fig.legend(handles=legend_handles(["No graph", "Averaging GNN", "High-pass GNN", "Contrast GNN"]),
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig1_synthetic")


# ------------------------------------------------------------------ fig2 METABRIC C-index
GNN_ARCHS = ["DeepSets", "GCN", "SAGE", "GAT", "GIN", "PNA", "FAGCN", "ACM-lite", "LC+GCN", "Contrast-GNN"]
RIDGE = {"Ridge: Mean (no graph)": "Ridge: mean", "Ridge: Mean+SD (no graph)": "Ridge: mean + SD",
         "Ridge: GRAD (graph contrast)": "Ridge: mean + SD + neighbour contrast",
         "Ridge: Clinical (reference)": "Ridge: clinical only"}
FAM2 = {**{a: ARCH_FAM[a] for a in GNN_ARCHS}, **{v: "Ridge Cox" for v in RIDGE.values()}}


def fig2():
    t = pd.read_csv(os.path.join(RES, "metabric", "cindex_table.csv"))
    t = t[t.endpoint == "DSS"]
    rows = GNN_ARCHS + list(RIDGE.values())
    tasks = [("all", "All patients (n = 689, 223 events)"), ("ER+", "ER+ (n = 527, 153 events)"),
             ("ER-", "ER- (n = 147, 65 events)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    for ax, (task, title) in zip(axes, tasks):
        q = t[t.task == task].set_index("model")
        ax.axvspan(0.3, q.null_95.max(), color="#EEEEEE", lw=0)
        ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
        for i, a in enumerate(rows):
            c = FAM[FAM2[a]]
            if a in GNN_ARCHS:
                for key, off, pl in [(a, -0.15, "mean"), (a + "+SDpool", 0.15, "sd")]:
                    r = q.loc[key]
                    ax.plot(r.C_ensemble, i + off, POOL_MARK[pl][0], color=c, mfc=c if pl == "sd" else "white",
                            ms=4.5, mew=1)
            else:
                key = [k for k, v in RIDGE.items() if v == a][0]
                ax.plot(q.loc[key].C_ensemble, i, "D", color=c, ms=4)
        clin = q.loc["Ridge: Clinical (reference)"].C_ensemble
        ax.axvline(clin, color="#000000", lw=0.6, ls=":")
        ax.set_xlim(0.45, 0.74)
        ax.set_xlabel("C-index, disease-specific survival")
        ax.set_title(title, fontsize=8)
        family_rows(ax, rows, FAM2)
    h = legend_handles(["No graph", "Averaging GNN", "High-pass GNN", "Contrast GNN"])
    h.append(Line2D([], [], ls="", marker="D", color="#000000", label="ridge Cox (no deep learning)"))
    h.append(plt.Rectangle((0, 0), 1, 1, color="#EEEEEE", label="95% of permutation null"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig2_metabric_cindex")


# ------------------------------------------------------------------ fig3 graph vs edge-removed twin
def boot_delta(t, e, A, B, n_boot=1000, seed=0):
    """Patient bootstrap of C(A) - C(B). A, B: (k, n) score arrays. Returns (k, n_boot)."""
    rng = np.random.default_rng(seed)
    n = len(t)
    out = np.empty((A.shape[0], n_boot))
    for b in range(n_boot):
        ii = rng.integers(0, n, n)
        tt, ee = t[ii], e[ii]
        i, j = np.where((ee[:, None] == 1) & (tt[:, None] < tt[None, :]))
        ai, aj, bi, bj = A[:, ii][:, i], A[:, ii][:, j], B[:, ii][:, i], B[:, ii][:, j]
        ca = ((ai > aj).sum(1) + 0.5 * (ai == aj).sum(1)) / len(i)
        cb = ((bi > bj).sum(1) + 0.5 * (bi == bj).sum(1)) / len(i)
        out[:, b] = ca - cb
    return out


def fig3():
    pts = pd.read_csv(os.path.join(ROOT, "data", "metabric", "processed", "patients.csv"))
    sc = np.load(os.path.join(RES, "metabric", "oof_scores_DSS.npz"))
    comp = pd.read_csv(os.path.join(RES, "metabric", "comparisons.csv"))
    comp = comp[comp.family == "PRIMARY graph vs no-graph"]
    graph = GNN_ARCHS[1:]
    pairs = [(a + pl, "DeepSets" + pl) for a in graph for pl in ("", "+SDpool")]
    A = np.stack([sc[a] for a, _ in pairs])  # npz keys equal the GNN model names
    B = np.stack([sc[b] for _, b in pairs])
    t, e = pts.dss_years.to_numpy(float), pts.dss_event.to_numpy(int)
    er = pts.ERStatus.to_numpy()
    tasks = [("all", "All patients", np.ones(len(pts), bool)), ("ER+", "ER+", er == "pos"), ("ER-", "ER-", er == "neg")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True, sharex=True)
    rows_out = []
    for ax, (task, title, m) in zip(axes, tasks):
        bd = boot_delta(t[m], e[m], A[:, m], B[:, m])
        lo, hi = np.quantile(bd, [0.025, 0.975], axis=1)
        q = comp[comp.task == task].set_index(["A", "B"])
        for k, (a, b) in enumerate(pairs):
            arch = a.replace("+SDpool", "")
            i = graph.index(arch)
            sd = a.endswith("+SDpool")
            off = 0.15 if sd else -0.15
            c = FAM[ARCH_FAM[arch]]
            d = q.loc[(a, b)].dC
            ax.plot([lo[k], hi[k]], [i + off] * 2, color=c, lw=1)
            ax.plot(d, i + off, "^" if sd else "o", color=c, mfc=c if sd else "white", ms=4.5, mew=1)
            rows_out.append({"task": task, "graph_model": a, "no_graph_twin": b, "dC": d, "ci_lo": lo[k],
                             "ci_hi": hi[k], "p": q.loc[(a, b)].p, "p_holm": q.loc[(a, b)].p_holm})
        ax.axvline(0, color="#000000", lw=0.8)
        pmin = q.p_holm.min()
        ax.set_title(f"{title}  (smallest Holm p = {pmin:.2f})", fontsize=8)
        family_rows(ax, graph, ARCH_FAM)
    fig.supxlabel("C-index of the GNN minus the same model with the edges removed (DeepSets), 95% bootstrap CI",
                  fontsize=8, y=0.04)
    fig.legend(handles=legend_handles(["Averaging GNN", "High-pass GNN", "Contrast GNN"]), loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig3_graph_vs_no_graph")
    pd.DataFrame(rows_out).round(4).to_csv(os.path.join(RES, "metabric", "graph_vs_no_graph_bootstrap_ci.csv"),
                                            index=False)


# ------------------------------------------------------------------ fig4 kernel
def fig4():
    s = pd.read_csv(os.path.join(RES, "kernel", "kernel_summary.csv"))
    e = pd.read_csv(os.path.join(RES, "kernel", "end_to_end.csv"))
    gpus = [("RTX PRO 6000 Blackwell", "#009E73"), ("H200 NVL", "#555555")]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    for gpu, c in gpus:
        for g, ls in [("METABRIC", "-"), ("METABRICx4", "--")]:
            q = s[(s.gpu == gpu) & (s.graph == g)]
            ax[0].plot(q.F, q.speedup_fwdbwd, ls, color=c, marker="o", ms=3.5,
                       label=f"{gpu.replace(' Blackwell', '')}, {'1.1 M' if g == 'METABRIC' else '4.4 M'} cells")
    q = s[s.gpu == "H200 NVL"]  # memory is identical on both GPUs
    for g, ls, lab in [("METABRIC", "-", "1.1 M cells"), ("METABRICx4", "--", "4.4 M cells")]:
        r = q[q.graph == g]
        ax[1].plot(r.F, r.PyG_mem_GB, ls, color="#D55E00", marker="o", ms=3.5, label=f"PyG, {lab}")
        ax[1].plot(r.F, r.Triton_mem_GB, ls, color="#0072B2", marker="s", ms=3.5, label=f"fused Triton, {lab}")
    for a in ax[:2]:
        a.set_xscale("log", base=2)
        a.set_xticks([16, 32, 64, 128])
        a.set_xticklabels([16, 32, 64, 128])
        a.set_xlabel("feature width F")
    ax[0].set_ylim(0)
    ax[0].set_ylabel("speedup vs PyG\n(forward + backward)")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("extra GPU memory (GB)")
    ax[1].set_ylim(0.08, 2000)
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    ax[1].set_title("same on both GPUs", fontsize=7)
    x = np.arange(len(gpus))
    for k, (impl, c) in enumerate([("PyG scatter", "#D55E00"), ("fused Triton", "#0072B2")]):
        v = [e[(e.gpu == g) & (e.impl == impl)].ms_per_step.iloc[0] for g, _ in gpus]
        ax[2].bar(x + (k - 0.5) * 0.38, v, 0.36, color=c, label="PyG" if k == 0 else "fused Triton")
        for xi, vi in zip(x, v):
            ax[2].text(xi + (k - 0.5) * 0.38, vi + 1, f"{vi:.0f}", ha="center", fontsize=6.5)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["RTX PRO 6000\nBlackwell", "H200 NVL"])
    ax[2].set_ylabel("ms per training step")
    ax[2].set_title("full Contrast-GNN training step", fontsize=7)
    ax[2].set_ylim(0, 75)
    ax[2].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", columnspacing=0.8, handlelength=1)
    fig.tight_layout()
    save(fig, "fig4_kernel")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
