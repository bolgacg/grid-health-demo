"""Detector architectures, in the shape of the centre's papers.
All trained on the healthy window only (faults are rare and unlabelled: Shaker 2026,
Energy & Buildings). Bootstrap ridge ensembles stand in for the deep models; the
uncertainty logic, not the architecture, is the subject."""
import numpy as np

def _feat_full(K, ambient):
    amb = np.broadcast_to(ambient, K.shape)
    return np.stack([np.ones_like(K), K, np.power(K, 1.6), amb, K*amb], axis=-1)

def _feat_lin(K, ambient):
    amb = np.broadcast_to(ambient, K.shape)
    return np.stack([np.ones_like(K), K, amb], axis=-1)

def _solve_batch(X, y, lam=1e-3):
    F = X.shape[-1]
    G = np.einsum('atf,atg->afg', X, X) + lam*np.eye(F)[None]
    b = np.einsum('atf,at->af', X, y)
    return np.linalg.solve(G, b[..., None])[..., 0]

def ensemble(K, ambient, hotspot, train_idx, m=15, seed=0, feat=_feat_full):
    """Bootstrap ensemble trained on train_idx days. Returns (mean pred, ensemble std)."""
    rng = np.random.default_rng(seed)
    X = feat(K, ambient)
    T = len(train_idx)
    preds = np.empty((m,) + hotspot.shape)
    for j in range(m):
        idx = train_idx[rng.integers(0, T, T)]
        w = _solve_batch(X[:, idx, :], hotspot[:, idx])
        preds[j] = np.einsum('adf,af->ad', X, w)
    return preds.mean(0), preds.std(0)

def zscore(resid, train_idx):
    mu = resid[:, train_idx].mean(1, keepdims=True)
    sd = resid[:, train_idx].std(1, keepdims=True) + 1e-9
    return (resid - mu) / sd

def ewma(z, lam=0.12):
    out = np.zeros_like(z)
    for t in range(1, z.shape[1]):
        out[:, t] = (1-lam)*out[:, t-1] + lam*z[:, t]
    return out

def cusum(z, k=0.75):
    out = np.zeros_like(z)
    for t in range(1, z.shape[1]):
        out[:, t] = np.maximum(0, out[:, t-1] + z[:, t] - k)
    return out

def tickets_from_alarm(alarm, refractory=10):
    out = {}
    A, D = alarm.shape
    for a in range(A):
        days, last = [], -10**9
        for t in np.flatnonzero(alarm[a]):
            if t - last >= refractory:
                days.append(int(t)); last = t
        out[a] = days
    return out

def score(tix, fault_assets, onset, fail_day, train_days, n_days, n_assets):
    tp, fp, warn, caught = 0, 0, [], 0
    for a, days in tix.items():
        for t in days:
            if t < train_days: continue
            if a in fault_assets and t >= onset[a] - 5: tp += 1
            else: fp += 1
    for a in fault_assets:
        fd = fail_day[a]
        hits = [t for t in tix.get(a, []) if onset[a] - 5 <= t <= (fd if fd else n_days)]
        if hits:
            caught += 1
            if fd: warn.append(fd - hits[0])
    months = (n_days - train_days) / 30.0
    return dict(detected=caught, of=len(fault_assets),
                warning_days=float(np.median(warn)) if warn else 0.0,
                fa_per_month=fp/months, precision=tp/(tp+fp) if tp+fp else 0.0,
                tickets=tp+fp)
