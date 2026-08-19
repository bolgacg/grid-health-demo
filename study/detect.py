"""Detection stack and scoring."""
import numpy as np

HPD = 24


def statistics(resid, warmup):
    """Three detection statistics computed on the residual stream."""
    mu = resid[:, :warmup].mean(1, keepdims=True)
    sd = resid[:, :warmup].std(1, keepdims=True) + 1e-9

    z = (resid - mu) / sd                                    # 1. standardised residual

    ew = np.zeros_like(resid)                                # 2. EWMA of residual
    lam = 0.02
    acc = np.zeros(resid.shape[0])
    for i in range(resid.shape[1]):
        acc = lam * z[:, i] + (1 - lam) * acc
        ew[:, i] = acc
    ew = ew / (np.sqrt(lam / (2 - lam)))

    cs = np.zeros_like(resid)                                # 3. one-sided CUSUM
    acc = np.zeros(resid.shape[0])
    slack = 0.5
    for i in range(resid.shape[1]):
        acc = np.maximum(0.0, acc + z[:, i] - slack)
        cs[:, i] = acc
    return z, ew, cs


def calibrate(resid, warmup, target_rate=2e-4):
    """Set each statistic's threshold from healthy data alone, so that each
    fires at the same small rate during the warmup window. This removes hand
    tuning and makes thresholds comparable across different systems."""
    z, ew, cs = statistics(resid, warmup)
    q = 1.0 - target_rate
    return (float(np.quantile(z[:, :warmup], q)),
            float(np.quantile(ew[:, :warmup], q)),
            float(np.quantile(cs[:, :warmup], q)))


def alarms(resid, unc, warmup, *, vote_k=2, thr=None,
           use_uncertainty=False, unc_q=0.98, persistence=1):
    """Returns a boolean alarm matrix.

    vote_k        how many of the three statistics must fire together
    use_uncertainty  abstain while ensemble disagreement is in its top (1-unc_q)
    persistence   require the vote to hold this many consecutive hours
    """
    if thr is None:
        thr = calibrate(resid, warmup)
    z, ew, cs = statistics(resid, warmup)
    votes = (z > thr[0]).astype(int) + (ew > thr[1]).astype(int) + (cs > thr[2]).astype(int)
    fired = votes >= vote_k

    if use_uncertainty:
        cut = np.quantile(unc[:, :warmup], unc_q, axis=1, keepdims=True)
        fired = fired & (unc <= cut)                         # abstain when unsure

    if persistence > 1:
        run = np.zeros(fired.shape[0], dtype=int)
        out = np.zeros_like(fired)
        for i in range(fired.shape[1]):
            run = np.where(fired[:, i], run + 1, 0)
            out[:, i] = run >= persistence
        fired = out

    fired[:, :warmup] = False
    return fired


def score(fired, faulty, fault_start, n_hours, warmup, refractory=24 * 7):
    """Operator-facing metrics.

    An alarm becomes a ticket; after a ticket the asset is silenced for
    `refractory` hours, so one developing problem is one ticket rather than a
    thousand. Without this, any detector that flickers looks worse than one
    that latches, which is an artefact of counting rather than a real effect.
    """
    tickets = 0
    healthy_hours = 0
    detected, delays = 0, []
    for a in range(fired.shape[0]):
        if faulty[a]:
            s_i = fault_start[a]
            tickets += _tickets(fired[a, warmup:s_i], refractory)
            healthy_hours += max(0, s_i - warmup)
            post = np.flatnonzero(fired[a, s_i:])
            if len(post):
                detected += 1
                delays.append(post[0] / 24.0)
        else:
            tickets += _tickets(fired[a, warmup:], refractory)
            healthy_hours += n_hours - warmup
    months = healthy_hours / (24 * 30.0)
    n_f = int(faulty.sum())
    true_tickets = detected
    precision = true_tickets / (true_tickets + tickets) if (true_tickets + tickets) else float("nan")
    return dict(
        false_alarms_per_asset_month=tickets / months if months else float("nan"),
        detection_rate=detected / n_f if n_f else float("nan"),
        median_delay_days=float(np.median(delays)) if delays else float("nan"),
        precision=precision,
        n_false=tickets, n_faulty=n_f, n_detected=detected)


def _tickets(mask, refractory):
    """Count alarm events, silencing an asset for `refractory` hours after each."""
    n = 0
    i = 0
    idx = np.flatnonzero(mask)
    for j in idx:
        if j >= i:
            n += 1
            i = j + refractory
    return n


def drift_onset(unc, warmup, k_days=3):
    """First hour at which mean ensemble disagreement stays above the warmup
    99th percentile for k_days running. This is the 'the world moved' signal,
    separate from any per-asset fault alarm."""
    m = unc.mean(0)
    cut = np.quantile(m[:warmup], 0.99)
    over = m > cut
    run = 0
    for i in range(warmup, len(m)):
        run = run + 1 if over[i] else 0
        if run >= k_days * 24:
            return i
    return None
