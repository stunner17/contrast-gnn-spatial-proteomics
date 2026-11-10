"""Minimal survival tools (no lifelines / sksurv needed): ridge Cox, Harrell's C-index, CV, nulls."""
import numpy as np
from scipy.optimize import minimize


def cindex(time, event, risk):
    """Harrell's C: among comparable pairs (i had the event and t_i < t_j), fraction with risk_i > risk_j."""
    t, e, r = map(np.asarray, (time, event, risk))
    ev = np.where(e == 1)[0]
    num = den = 0.0
    for i in ev:
        comp = t > t[i]
        den += comp.sum()
        num += (r[i] > r[comp]).sum() + 0.5 * (r[i] == r[comp]).sum()
    return num / den if den > 0 else np.nan


def _negloglik(beta, X, t, e, lam):
    order = np.argsort(-t)
    Xo, eo = X[order], e[order]
    eta = Xo @ beta
    m = eta.max()
    log_cum = np.log(np.cumsum(np.exp(eta - m))) + m  # risk set = all with time >= t_i (Breslow)
    ll = np.sum(eo * (eta - log_cum))
    return -ll + 0.5 * lam * beta @ beta


def _grad(beta, X, t, e, lam):
    order = np.argsort(-t)
    Xo, eo = X[order], e[order]
    eta = Xo @ beta
    m = eta.max()
    w = np.exp(eta - m)
    cw = np.cumsum(w)
    cwx = np.cumsum(w[:, None] * Xo, axis=0)
    g = np.sum(eo[:, None] * (Xo - cwx / cw[:, None]), axis=0)
    return -g + lam * beta


class RidgeCox:
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, t, e):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        b0 = np.zeros(Z.shape[1])
        res = minimize(_negloglik, b0, jac=_grad, args=(Z, t, e.astype(float), self.lam),
                       method="L-BFGS-B")
        self.beta = res.x
        return self

    def risk(self, X):
        return ((X - self.mu) / self.sd) @ self.beta


def strat_folds(event, k, rng):
    """Folds stratified by event status."""
    idx = np.arange(len(event))
    folds = [[] for _ in range(k)]
    for val in (0, 1):
        ii = rng.permutation(idx[event == val])
        for j, i in enumerate(ii):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds]


def cv_cox(X, t, e, k=5, repeats=10, lams=(0.1, 1, 10, 100), seed=0, inner_k=3):
    """Repeated k-fold CV with an inner CV to choose the ridge penalty.
    Returns per-repeat C-index and the repeat-averaged out-of-fold risk (rank-normalised)."""
    rng = np.random.default_rng(seed)
    cis, oof_sum = [], np.zeros(len(t))
    for _ in range(repeats):
        oof = np.zeros(len(t))
        for te in strat_folds(e, k, rng):
            tr = np.setdiff1d(np.arange(len(t)), te)
            best, best_c = lams[0], -1
            if len(lams) > 1:
                inner = strat_folds(e[tr], inner_k, rng)
                for lam in lams:
                    cs = []
                    for ite in inner:
                        itr = np.setdiff1d(np.arange(len(tr)), ite)
                        m = RidgeCox(lam).fit(X[tr][itr], t[tr][itr], e[tr][itr])
                        cs.append(cindex(t[tr][ite], e[tr][ite], m.risk(X[tr][ite])))
                    if np.mean(cs) > best_c:
                        best, best_c = lam, np.mean(cs)
            m = RidgeCox(best).fit(X[tr], t[tr], e[tr])
            # rank-normalise within the test fold so folds are comparable when pooled
            r = m.risk(X[te])
            oof[te] = (np.argsort(np.argsort(r)) + 0.5) / len(r)
        cis.append(cindex(t, e, oof))
        oof_sum += oof
    return np.array(cis), oof_sum / repeats


def perm_null_cv(X, t, e, n_perm=200, seed=123, **kw):
    """Null C-index distribution: permute (time, event) pairs across patients, rerun the whole CV."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        p = rng.permutation(len(t))
        c, _ = cv_cox(X, t[p], e[p], repeats=1, lams=(10,), seed=int(rng.integers(1e9)), **kw)
        null.append(c.mean())
    return np.array(null)


def paired_perm_cindex(t, e, r1, r2, n=2000, seed=1):
    """One-sided test that risk r1 ranks better than r2: swap the two scores per patient."""
    rng = np.random.default_rng(seed)
    obs = cindex(t, e, r1) - cindex(t, e, r2)
    cnt = 0
    for _ in range(n):
        s = rng.random(len(t)) < 0.5
        cnt += (cindex(t, e, np.where(s, r2, r1)) - cindex(t, e, np.where(s, r1, r2))) >= obs
    return obs, (1 + cnt) / (n + 1)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty_like(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj
