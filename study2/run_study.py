"""Runs the detector studies across scenarios, seeds and alarm budgets; emits
results.json (verdicts) and timeline.json (browser payload). Seeded, reproducible."""
import json, sys, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
import sim, detect, survival

SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [3.6, 3.2, 2.9, 2.6, 2.3, 2.0]   # z-units threshold ladder (the alarm-budget slider)
DEFAULT_B = 2.9

def build_stats(s, seed):
    tr = np.arange(sim.TRAIN_DAYS)
    pred, unc = detect.ensemble(s['K'], s['ambient'], s['hotspot'], tr, seed=seed)
    pred2, _  = detect.ensemble(s['K'], s['ambient'], s['hotspot'], tr, seed=seed+100,
                                feat=detect._feat_lin, m=7)
    z  = detect.zscore(s['hotspot'] - pred,  tr)
    z2 = detect.zscore(s['hotspot'] - pred2, tr)
    zew = detect.ewma(z)
    cus = detect.cusum(z)
    unc_hi = unc > np.quantile(unc[:, tr], 0.995)     # inputs left the training envelope
    return dict(z=z, z2=z2, zew=zew, cus=cus, unc=unc, unc_hi=unc_hi, pred=pred)

def recalibrated_z(s, seed, window=30, every=7):
    """Rolling refit: every 7 days retrain on the trailing 30 days (labels unknown, so the
    window absorbs whatever is in it, faults included). Returns the recalibrated z AND the
    redeployment day list: each redeploy restarts the monitoring statistic, as it does in
    practice (new model, new baseline, new accumulator)."""
    z = np.zeros_like(s['hotspot'])
    seg_starts = list(range(sim.TRAIN_DAYS, sim.N_DAYS, every))
    for t0 in seg_starts:
        tr = np.arange(max(0, t0-window), t0)
        pred, _ = detect.ensemble(s['K'], s['ambient'], s['hotspot'], tr, seed=seed, m=7)
        seg = slice(t0, min(t0+every, sim.N_DAYS))
        mu = (s['hotspot']-pred)[:, tr].mean(1, keepdims=True)
        sd = (s['hotspot']-pred)[:, tr].std(1, keepdims=True) + 1e-9
        z[:, seg] = ((s['hotspot']-pred)[:, seg] - mu) / sd
    return z, seg_starts

def segmented_cusum(z, seg_starts, k=0.75):
    """CUSUM restarted at every redeployment."""
    out = np.zeros_like(z)
    starts = set(seg_starts)
    for t in range(1, z.shape[1]):
        prev = 0.0 if t in starts else out[:, t-1]
        out[:, t] = np.maximum(0, prev + z[:, t] - k)
    return out

def alarms(st, b, s):
    """The five architectures at threshold b (z-units)."""
    lim = np.quantile(s['hotspot'][:, :sim.TRAIN_DAYS], 0.997) # limit set at commissioning
    naive  = s['hotspot'] > lim
    single = st['z'] > b
    vote   = ((st['z'] > b).astype(int) + (st['z2'] > b).astype(int)
              + (st['zew'] > 0.55*b).astype(int)) >= 2
    # uncertainty gate: when the ensemble disagrees, demand a unanimous vote (weighting, not veto)
    vote3  = ((st['z'] > b).astype(int) + (st['z2'] > b).astype(int)
              + (st['zew'] > 0.55*b).astype(int)) >= 3
    gated  = np.where(st['unc_hi'], vote3, vote)
    seq    = gated & (st['cus'] > 3.0*b)               # persistence: sustained evidence only
    return dict(naive=naive, single=single, vote=vote, gated=gated, seq=seq)

ARCH = dict(naive='static limit', single='forecaster residual', vote='multi-method vote',
            gated='vote + uncertainty weighting', seq='full sequential gate')

def main():
    results = {}
    keep_tl = {}
    for scenario in ['stationary', 'shift']:
        agg = {a: {b: [] for b in BUDGETS} for a in list(ARCH)+ (['recal'] if scenario=='shift' else [])}
        for seed in SEEDS:
            s = sim.simulate(scenario, seed)
            st = build_stats(s, seed)
            if scenario == 'shift':
                zr, segs = recalibrated_z(s, seed)
                cus_r = segmented_cusum(zr, segs)
            for b in BUDGETS:
                mats = alarms(st, b, s)
                if scenario == 'shift':
                    mats['recal'] = cus_r > 3.0*b
                for arch, alarm in mats.items():
                    tix = detect.tickets_from_alarm(alarm)
                    agg[arch][b].append(detect.score(tix, s['fault_assets'], s['onset'],
                        s['fail_day'], sim.TRAIN_DAYS, sim.N_DAYS, sim.N_ASSETS))
            if seed == 0:
                keep_tl[scenario] = (s, st, (zr if scenario=='shift' else None),
                                     (segs if scenario=='shift' else None))
        label = dict(ARCH, recal='sequential gate + rolling recalibration')
        results[scenario] = {label[a]: {str(b): {k: round(float(np.mean([r[k] for r in runs])),3)
                             for k in runs[0]} for b, runs in per_b.items()}
                             for a, per_b in agg.items()}
    results['cables'] = survival.run()
    results['meta'] = dict(seeds=SEEDS, budgets=BUDGETS, default_budget=DEFAULT_B,
                           n_assets=sim.N_ASSETS, n_days=sim.N_DAYS,
                           train_days=sim.TRAIN_DAYS, regime_day=sim.REGIME_DAY)
    base = __file__.rsplit('/',1)[0]
    json.dump(results, open(base+'/results.json','w'), indent=1)

    # ---- browser payload: seed-0 timelines + per-budget tickets ----
    tl = dict(meta=results['meta'])
    for scenario, (s, st, zr, zsegs) in keep_tl.items():
        tickets = {}
        for b in BUDGETS:
            mats = alarms(st, b, s)
            if zr is not None: mats['recal'] = segmented_cusum(zr, zsegs) > 3.0*b
            tickets[str(b)] = {a: detect.tickets_from_alarm(m) for a, m in mats.items()}
        tl[scenario] = dict(
            ambient=[round(float(x),1) for x in s['ambient']],
            K=[[round(float(x),2) for x in row] for row in s['K']],
            hotspot=[[round(float(x),1) for x in row] for row in s['hotspot']],
            pred=[[round(float(x),1) for x in row] for row in st['pred']],
            z=[[round(float(x),2) for x in row] for row in st['z']],
            cus=[[round(float(x),1) for x in row] for row in st['cus']],
            unc=[[round(float(x),2) for x in row] for row in st['unc']],
            unc_hi=[[int(x) for x in row] for row in st['unc_hi']],
            ramp=[[round(float(x),3) for x in row] for row in s['ramp']],
            fault_assets=s['fault_assets'],
            onset={str(k):v for k,v in s['onset'].items()},
            fail_day={str(k):v for k,v in s['fail_day'].items()},
            tickets=tickets)
    json.dump(tl, open(base+'/timeline.json','w'), separators=(',',':'))

    B = str(DEFAULT_B)
    for scenario in ['stationary','shift']:
        print(f'== {scenario} (threshold z={B}) ==')
        print(f'{"architecture":38s} {"det":>5s} {"warn(d)":>8s} {"FA/mo":>7s} {"prec":>6s}')
        for arch, per_b in results[scenario].items():
            r = per_b[B]
            print(f'{arch:38s} {r["detected"]:5.1f} {r["warning_days"]:8.1f} {r["fa_per_month"]:7.2f} {r["precision"]:6.2f}')
    c = results['cables']
    print('cables: naive', round(c['naive_mean'],1), 'KM', round(c['km_median'],1),
          '| oldest', c['rule_oldest'], '| hazard', c['rule_hazard'])
    import os
    print('timeline.json bytes:', os.path.getsize(base+'/timeline.json'))

if __name__ == '__main__':
    main()
