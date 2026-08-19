import json, sys
import numpy as np
sys.path.insert(0, "/home/bolgac/projects/grid-health-demo/study")
from sim import system_transformer, system_districtheating, RidgeEnsemble
from detect import alarms, score, drift_onset

HPD = 24
N_HOURS = 24 * 300
WARMUP = 24 * 90          # model is fitted on this healthy window
N_ASSETS = 120
FAULT_FRAC = 0.25

CONFIGS = {
    "single threshold":       dict(vote_k=1, use_uncertainty=False, persistence=1),
    "multi-method vote":      dict(vote_k=2, use_uncertainty=False, persistence=1),
    "vote + persistence":     dict(vote_k=2, use_uncertainty=False, persistence=6),
    "vote + uncertainty":     dict(vote_k=2, use_uncertainty=True,  persistence=1),
    "full sequential gate":   dict(vote_k=2, use_uncertainty=True,  persistence=6),
}


def run(system_fn, shift, seed):
    rng = np.random.default_rng(seed)
    shift_at = int(N_HOURS * 0.55) if shift else None
    sysd = system_fn(rng, N_HOURS, N_ASSETS, FAULT_FRAC, shift_at=shift_at,
                     shift_load=1.28 if shift else 1.0,
                     shift_ambient=6.0 if shift else 0.0)
    X, y = sysd["X"], sysd["y"]

    # fit only on the healthy warmup window of healthy assets
    healthy = ~sysd["faulty"]
    ens = RidgeEnsemble(k=8, seed=seed).fit(X[healthy, :WARMUP], y[healthy, :WARMUP])
    mu, sd = ens.predict(X)
    resid = y - mu

    out = {}
    for name, cfg in CONFIGS.items():
        fired = alarms(resid, sd, WARMUP, **cfg)
        out[name] = score(fired, sysd["faulty"], sysd["fault_start"], N_HOURS, WARMUP)

    # --- recalibration arm: notice the regime change, refit, carry on ---
    onset = drift_onset(sd, WARMUP)
    if onset is not None:
        settle = onset + 14 * HPD                      # let the new regime settle
        if settle + 30 * HPD < N_HOURS:
            ens2 = RidgeEnsemble(k=8, seed=seed + 99).fit(
                X[healthy, settle:settle + 30 * HPD], y[healthy, settle:settle + 30 * HPD])
            mu2, sd2 = ens2.predict(X)
            resid2 = np.concatenate([resid[:, :settle], (y - mu2)[:, settle:]], axis=1)
            sdc = np.concatenate([sd[:, :settle], sd2[:, settle:]], axis=1)
            fired = alarms(resid2, sdc, WARMUP, vote_k=2, use_uncertainty=True, persistence=6)
            fired[:, settle:settle + 30 * HPD] = False      # blind during refit
            out["gate + recalibration"] = score(
                fired, sysd["faulty"], sysd["fault_start"], N_HOURS, WARMUP)
            out["gate + recalibration"]["drift_detected_day"] = float((onset - WARMUP) / 24.0)
    return out


def main():
    systems = {"transformer": system_transformer, "districtheating": system_districtheating}
    results = {}
    for sname, fn in systems.items():
        for shift in (False, True):
            key = f"{sname}/{'shift' if shift else 'stationary'}"
            runs = [run(fn, shift, seed) for seed in range(5)]
            agg = {}
            names = list(CONFIGS) + (["gate + recalibration"] if "gate + recalibration" in runs[0] else [])
            for cfg in names:
                agg[cfg] = {
                    m: float(np.nanmean([r[cfg][m] for r in runs if cfg in r]))
                    for m in ("false_alarms_per_asset_month", "detection_rate",
                              "median_delay_days", "precision")
                }
            results[key] = agg
            print(f"\n=== {key} ===")
            print(f"{'config':<24}{'FA/asset-mo':>13}{'detect':>9}{'delay(d)':>10}{'precision':>11}")
            for cfg, v in agg.items():
                print(f"{cfg:<24}{v['false_alarms_per_asset_month']:>13.2f}"
                      f"{v['detection_rate']:>9.2f}{v['median_delay_days']:>10.1f}{v['precision']:>11.2f}")
    json.dump(results, open("/home/bolgac/projects/grid-health-demo/study/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
