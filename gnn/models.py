"""GNN survival models for patient-level Cox regression on cell graphs.

Every model: Linear encoder -> 2 graph layers -> pool (mean, or mean+SD) -> BatchNorm on the pooled
vector -> MLP -> one risk score per image. BatchNorm on the pooled vector is needed: without it the
between-image differences after pooling are tiny and some models never start learning (found in an early pilot).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (FAConv, GATConv, GCNConv, GINConv, PNAConv, SAGEConv,
                                global_mean_pool)
from torch_geometric.utils import scatter

BASE_MODELS = ["DeepSets", "GCN", "SAGE", "GAT", "GIN", "PNA", "FAGCN", "ACM-lite",
               "LC+GCN", "Contrast-GNN"]
SDPOOL = "+SDpool"
ALL_MODELS = BASE_MODELS + [m + SDPOOL for m in BASE_MODELS]


def split_model(name):
    return (name[:-len(SDPOOL)], "meansd") if name.endswith(SDPOOL) else (name, "mean")


def mean_neighbors(x, ei):
    return scatter(x[ei[0]], ei[1], dim=0, dim_size=x.size(0), reduce="mean")


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


class SurvNet(nn.Module):
    def __init__(self, name, d_in, d=32, n_layers=2, deg_hist=None, dropout=0.1):
        super().__init__()
        kind, pool = split_model(name)
        self.kind, self.pool, self.dropout = kind, pool, dropout
        self.enc = nn.Linear(2 * d_in if kind == "LC+GCN" else d_in, d)
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
                c = PNAConv(d, d, aggregators=["mean", "min", "max", "std"], scalers=["identity"], deg=deg_hist)
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
        self.pool_norm = nn.BatchNorm1d(d_pool)
        self.head = nn.Sequential(nn.Linear(d_pool, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x, ei, batch, n_graphs):
        if self.kind == "LC+GCN":
            x = torch.cat([x, (x - mean_neighbors(x, ei)).abs()], dim=1)
        h = F.relu(self.enc(x))
        h0 = h
        for c in self.convs:
            if c is None:
                continue
            h = F.relu(c(h, h0, ei)) if self.kind == "FAGCN" else F.relu(c(h, ei))
        z = global_mean_pool(h, batch, size=n_graphs)
        if self.pool == "meansd":
            sd = (global_mean_pool(h * h, batch, size=n_graphs) - z * z).clamp_min(1e-8).sqrt()
            z = torch.cat([z, sd], dim=1)
        return self.head(self.pool_norm(z)).squeeze(-1)


def cox_loss(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow), mean over events."""
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_cum = torch.logcumsumexp(r, dim=0)
    return -((r - log_cum) * e).sum() / e.sum().clamp_min(1)
