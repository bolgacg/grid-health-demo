"""Simulated 10 kV distribution network: 1 primary substation, 4 feeders, 24 distribution
transformers. Daily resolution, 300 days. Transformer thermal model is a simplified IEC
60076-7 shape: hotspot = ambient + a*K^1.6, with a developing cooling fault raising the
thermal gain 'a' along a logistic ramp. Seeded, reproducible."""
import numpy as np

N_ASSETS = 24
N_DAYS = 300
TRAIN_DAYS = 60           # healthy calibration window
REGIME_DAY = 180          # heat wave + EV uptake start (shift scenario only)
FAULTS = [                # (asset, onset_day, ramp_days, gain_boost)
    (4,  110, 55, 0.42),   # feeder 1, cooling degradation over ~8 weeks
    (13, 140, 40, 0.50),   # feeder 3, faster
    (21, 130, 160, 0.26),  # feeder 4, slow creeper: months of gentle decay across the regime change
]
FAIL_MARGIN = 0.80        # fault counts as "failure imminent" when ramp passes 80%

def simulate(scenario, seed):
    rng = np.random.default_rng(seed)
    days = np.arange(N_DAYS)
    # shared ambient: Danish-ish summer->autumn, deg C
    ambient = 16 + 6*np.sin(2*np.pi*(days-30)/365.0) + rng.normal(0, 1.6, N_DAYS)
    if scenario == "shift":
        hw = (days >= REGIME_DAY) & (days < REGIME_DAY+25)
        ambient = ambient + np.where(hw, 6.5*np.exp(-((days-REGIME_DAY-9)/9.0)**2), 0)
    # per-transformer utilisation K (per-unit of rating)
    base = rng.uniform(0.52, 0.78, N_ASSETS)
    weekly = 0.06*np.sin(2*np.pi*days/7.0 + rng.uniform(0, 6.28, (N_ASSETS,1)))
    seasonal = 0.05*np.sin(2*np.pi*(days-40)/365.0)
    K = base[:,None] + weekly + seasonal + rng.normal(0, 0.035, (N_ASSETS, N_DAYS))
    if scenario == "shift":
        ev = np.clip((days - REGIME_DAY)/90.0, 0, 1) * rng.uniform(0.10, 0.22, N_ASSETS)[:,None]
        K = K + np.where(days >= REGIME_DAY, ev, 0)
    K = np.clip(K, 0.25, 1.35)
    # thermal gain a: nominal per asset, plus fault ramp
    a0 = rng.uniform(46, 54, N_ASSETS)
    ramp = np.zeros((N_ASSETS, N_DAYS))
    for asset, onset, rdays, boost in FAULTS:
        ramp[asset] = boost / (1 + np.exp(-(days - onset - rdays/2)/(rdays/7.5)))
    a = a0[:,None] * (1 + ramp)
    hotspot = ambient[None,:] + a*np.power(K, 1.6) + rng.normal(0, 1.1, (N_ASSETS, N_DAYS))
    fail_day = {}
    for asset, onset, rdays, boost in FAULTS:
        crossed = np.where(ramp[asset] >= FAIL_MARGIN*boost)[0]
        fail_day[asset] = int(crossed[0]) if len(crossed) else None
    return dict(ambient=ambient, K=K, hotspot=hotspot, ramp=ramp,
                fault_assets=[f[0] for f in FAULTS], fail_day=fail_day,
                onset={f[0]: f[1] for f in FAULTS})
