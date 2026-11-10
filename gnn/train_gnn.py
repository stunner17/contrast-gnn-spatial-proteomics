"""GPU: train GNN survival models on METABRIC IMC cell graphs.

Task = (endpoint, model, outer fold, seed). For each task:
  - outer fold f is the test set (fixed patient folds in folds.csv, made by scripts/02b_build_metabric.py, stratified by DSS event);
  - 20% of the remaining patients (stratified by event, seeded) is an inner validation set;
  - inputs are z-scored with training-cell statistics;
  - full-batch training on the GPU (all training images in one batch, Cox loss over images, each
    image carries its patient's time/event), Adam, up to --max-epochs, early stopping on the
    validation patient-level C-index (eval every 5 epochs, patience --patience epochs);
  - test risk per patient = mean risk over that patient's images, from the best-validation weights.
Each task writes preds/<endpoint>_<model>_f<fold>_s<seed>.npz; finished tasks are skipped (resumable).
Several processes share one GPU via --slice k/n (see run_*.sbatch).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import degree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ALL_MODELS, SurvNet, cox_loss  # noqa: E402

T0 = time.time()


def cindex(t, e, r):
    t, e, r = map(np.asarray, (t, e, r))
    num = den = 0.0
    for i in np.where(e == 1)[0]:
        comp = t > t[i]
        den += comp.sum()
        num += (r[i] > r[comp]).sum() + 0.5 * (r[i] == r[comp]).sum()
    return num / den if den else np.nan


def load(data_dir, endpoint, dev):
    graphs = torch.load(os.path.join(data_dir, "graphs.pt"))
    pts = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    folds = pd.read_csv(os.path.join(data_dir, "folds.csv"))
    pts = pts.merge(folds, on="metabric_id")
    if endpoint == "DSS":
        pts["t"], pts["e"] = pts.dss_years, pts.dss_event
    else:
        pts = pts[pts.os_months.notna()].copy()
        pts["t"], pts["e"] = pts.os_months, pts.os_event
    pts = pts.reset_index(drop=True)
    pidx = {p: i for i, p in enumerate(pts.metabric_id)}
    graphs = [g for g in graphs if g["patient"] in pidx]
    # concatenate every image into one big graph on the GPU
    xs, eis, gid, off = [], [], [], 0
    for k, g in enumerate(graphs):
        n = g["x"].shape[0]
        xs.append(g["x"].float())
        eis.append(g["edge_index"].long() + off)
        gid.append(torch.full((n,), k, dtype=torch.long))
        off += n
    D = {"x": torch.cat(xs).to(dev), "ei": torch.cat(eis, 1).to(dev), "gid": torch.cat(gid).to(dev),
         "img_patient": np.array([pidx[g["patient"]] for g in graphs]), "pts": pts,
         "deg": degree(torch.cat(eis, 1)[1], off, dtype=torch.long)}
    return D


def subgraph(D, img_mask_np, dev):
    """Node-induced subgraph made of the selected images, with re-indexed nodes and graph ids."""
    img_mask = torch.as_tensor(img_mask_np, device=dev)
    node_mask = img_mask[D["gid"]]
    new_node = torch.full_like(D["gid"], -1)
    new_node[node_mask] = torch.arange(int(node_mask.sum()), device=dev)
    keep_e = node_mask[D["ei"][0]] & node_mask[D["ei"][1]]
    ei = new_node[D["ei"][:, keep_e]]
    new_img = torch.full((len(img_mask_np),), -1, dtype=torch.long, device=dev)
    sel = torch.where(img_mask)[0]
    new_img[sel] = torch.arange(len(sel), device=dev)
    return D["x"][node_mask], ei, new_img[D["gid"][node_mask]], len(sel), node_mask


def run_task(D, endpoint, model, fold, seed, args, dev, out_file):
    t_start = time.time()
    pts = D["pts"]
    rng = np.random.default_rng(1000 * fold + seed)
    test_p = np.where(pts.fold.values == fold)[0]
    rest = np.where(pts.fold.values != fold)[0]
    val_p = np.concatenate([rng.choice(rest[pts.e.values[rest] == v],
                                       size=int(round(0.2 * (pts.e.values[rest] == v).sum())), replace=False)
                            for v in (0, 1)])
    train_p = np.setdiff1d(rest, val_p)
    ip = D["img_patient"]
    masks = {s: np.isin(ip, p) for s, p in (("train", train_p), ("val", val_p), ("test", test_p))}
    parts = {s: subgraph(D, m, dev) for s, m in masks.items()}
    mu = parts["train"][0].mean(0)
    sd = parts["train"][0].std(0) + 1e-6
    T = torch.tensor(pts.t.values, dtype=torch.float32, device=dev)
    E = torch.tensor(pts.e.values, dtype=torch.float32, device=dev)
    img_pat = {s: torch.as_tensor(ip[m], device=dev) for s, m in masks.items()}

    torch.manual_seed(seed)
    deg_hist = None
    if model.startswith("PNA"):
        d = D["deg"][parts["train"][4].cpu()]
        deg_hist = torch.bincount(d, minlength=int(d.max()) + 1)
    net = SurvNet(model, d_in=D["x"].shape[1], deg_hist=deg_hist).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def predict(split):
        x, ei, b, ng, _ = parts[split]
        net.eval()
        with torch.no_grad():
            r = net((x - mu) / sd, ei, b, ng)
        pid = img_pat[split]
        pr = torch.zeros(len(pts), device=dev).index_add_(0, pid, r)
        cnt = torch.zeros(len(pts), device=dev).index_add_(0, pid, torch.ones_like(r))
        keep = cnt > 0
        return torch.where(keep)[0].cpu().numpy(), (pr[keep] / cnt[keep]).cpu().numpy()

    x, ei, b, ng, _ = parts["train"]
    xz = (x - mu) / sd
    tt, ee = T[img_pat["train"]], E[img_pat["train"]]
    best, best_ep, best_state, since, t_train = -1.0, 0, None, 0, 0.0
    for ep in range(1, args.max_epochs + 1):
        t1 = time.time()
        net.train()
        opt.zero_grad()
        loss = cox_loss(net(xz, ei, b, ng), tt, ee)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t1
        if ep % 5 == 0:
            vi, vr = predict("val")
            c = cindex(pts.t.values[vi], pts.e.values[vi], vr)
            if c > best:
                best, best_ep, since = c, ep, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 5
            if since >= args.patience:
                break
    net.load_state_dict(best_state)
    ti, tr = predict("test")
    c_test = cindex(pts.t.values[ti], pts.e.values[ti], tr)
    np.savez(out_file, patient=pts.metabric_id.values[ti], risk=tr, time=pts.t.values[ti], event=pts.e.values[ti],
             best_epoch=best_ep, val_c=best, test_c=c_test, epochs_run=ep,
             sec_train=t_train, sec_total=time.time() - t_start)
    return {"best_epoch": best_ep, "epochs_run": ep, "val_c": best, "test_c": c_test,
            "sec_per_epoch": t_train / ep, "sec_total": time.time() - t_start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metabric/processed")
    ap.add_argument("--out", default="results/02_gnn")
    ap.add_argument("--endpoints", default="DSS,OS")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--slice", default="0/1", help="k/n: this process runs tasks k, k+n, k+2n, ...")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--time-limit-h", type=float, default=7.4)
    args = ap.parse_args()
    k, n = map(int, args.slice.split("/"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    pred_dir = os.path.join(args.out, "preds")
    os.makedirs(pred_dir, exist_ok=True)
    tasks = [(ep, m, f, s) for ep in args.endpoints.split(",") for m in args.models.split(",")
             for f in map(int, args.folds.split(",")) for s in map(int, args.seeds.split(","))]
    mine = tasks[k::n]
    print(f"[slice {k}/{n}] device {dev} ({torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu'}); "
          f"{len(mine)} of {len(tasks)} tasks", flush=True)
    data = {}
    for i, (ep, m, f, s) in enumerate(mine):
        out_file = os.path.join(pred_dir, f"{ep}_{m.replace('+', 'p')}_f{f}_s{s}.npz")
        if os.path.exists(out_file):
            continue
        if (time.time() - T0) / 3600 > args.time_limit_h:
            print(f"[slice {k}/{n}] time limit reached; {len(mine) - i} tasks left, resubmit to continue", flush=True)
            return
        if ep not in data:
            t1 = time.time()
            data[ep] = load(args.data, ep, dev)
            print(f"[slice {k}/{n}] loaded {ep}: {data[ep]['x'].shape[0]} cells, {data[ep]['ei'].shape[1]} edges, "
                  f"{len(data[ep]['pts'])} patients in {time.time() - t1:.1f}s", flush=True)
        r = run_task(data[ep], ep, m, f, s, args, dev, out_file)
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0
        print(f"[slice {k}/{n}] {ep} {m:22s} f{f} s{s}: val C {r['val_c']:.3f} test C {r['test_c']:.3f} "
              f"best ep {r['best_epoch']}/{r['epochs_run']}  {r['sec_per_epoch'] * 1000:.0f} ms/epoch  "
              f"{r['sec_total']:.0f}s  peak GPU mem {mem:.1f} GB", flush=True)
    print(f"[slice {k}/{n}] all my tasks done", flush=True)


if __name__ == "__main__":
    main()
