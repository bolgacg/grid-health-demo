"""
Grid-health study: fault detection on two simulated energy systems, with and
without uncertainty gating, evaluated under distribution shift.

Two systems:
  A) LV distribution transformers  - incipient thermal degradation
  B) District heating substations  - heat-exchanger fouling

Detector stack (mirrors the SDU Center for Energy Informatics literature):
  residual from an ensemble regressor  ->  several detection statistics  ->
  vote  ->  optional uncertainty abstention  ->  optional persistence gate
"""
import numpy as np

HOURS_PER_DAY = 24


# ----------------------------------------------------------------- systems

def _daily_shape(rng, n, peak_hours=(8, 19), amp=1.0):
    t = np.arange(n)
    h = t % HOURS_PER_DAY
    dow = (t // HOURS_PER_DAY) % 7
    base = 0.55 + 0.45 * np.exp(-0.5 * ((h - peak_hours[0]) / 3.0) ** 2) \
                + 0.45 * np.exp(-0.5 * ((h - peak_hours[1]) / 2.5) ** 2)
    weekend = np.where(dow >= 5, 0.80, 1.0)
    return amp * base * weekend


def system_transformer(rng, n_hours, n_assets, fault_frac, shift_at=None,
                       shift_load=1.0, shift_ambient=0.0):
    """LV distribution transformer fleet.

    Observable: top-oil temperature rise over ambient.
    Physics: dTheta/dt = (k * loading^2 - Theta) / tau   (simple thermal lag)
    Fault: cooling degradation -> gain k drifts up slowly (ramp over ~30 days).
    """
    t = np.arange(n_hours)
    ambient = 9.0 + 7.0 * np.sin(2 * np.pi * (t - 2000) / (365 * HOURS_PER_DAY)) \
                  + 3.0 * np.sin(2 * np.pi * (t % HOURS_PER_DAY - 4) / HOURS_PER_DAY)
    load = np.zeros((n_assets, n_hours))
    theta = np.zeros((n_assets, n_hours))
    faulty = rng.random(n_assets) < fault_frac
    fault_start = np.where(faulty, rng.integers(int(n_hours * 0.45), int(n_hours * 0.80), n_assets), -1)

    tau = 3.0
    rated = rng.uniform(0.45, 0.75, (n_assets, 1))
    shape = np.stack([_daily_shape(rng, n_hours) for _ in range(1)])[0]
    load = np.clip(rated * shape + rng.normal(0, 0.035, (n_assets, n_hours)), 0.02, 1.35)
    if shift_at is not None:
        load[:, shift_at:] *= shift_load

    k = np.full((n_assets, n_hours), 55.0)
    ramp = np.clip((t[None, :] - fault_start[:, None]) / (30 * HOURS_PER_DAY), 0, 1)
    k = np.where(faulty[:, None], k * (1 + 0.16 * ramp), k)

    drive = k * load ** 2
    theta = np.zeros((n_assets, n_hours))
    for i in range(1, n_hours):
        theta[:, i] = theta[:, i - 1] + (drive[:, i] - theta[:, i - 1]) / tau
    theta += rng.normal(0, 0.9, (n_assets, n_hours))

    amb = np.tile(ambient, (n_assets, 1))
    if shift_at is not None:
        amb[:, shift_at:] += shift_ambient
    return dict(name="transformer", X=np.stack([load, amb / 30.0], -1),
                y=theta, faulty=faulty, fault_start=fault_start)


def system_districtheating(rng, n_hours, n_assets, fault_frac, shift_at=None,
                           shift_load=1.0, shift_ambient=0.0):
    """District-heating substation fleet.

    Observable: return temperature.
    Physics: good heat exchange -> low return temp; return rises with flow
             and falls with available supply-return spread.
    Fault: fouling -> effectiveness decays -> return temperature creeps up.
    """
    t = np.arange(n_hours)
    ambient = 9.0 + 7.0 * np.sin(2 * np.pi * (t - 2000) / (365 * HOURS_PER_DAY))
    supply = 75.0 - 0.45 * ambient                       # weather-compensated
    heat = np.zeros((n_assets, n_hours))
    ret = np.zeros((n_assets, n_hours))
    faulty = rng.random(n_assets) < fault_frac
    fault_start = np.where(faulty, rng.integers(int(n_hours * 0.45), int(n_hours * 0.80), n_assets), -1)

    size = rng.uniform(0.5, 1.2, (n_assets, 1))
    shape = _daily_shape(rng, n_hours, peak_hours=(7, 20))
    demand = size * shape * np.clip((18.0 - ambient) / 12.0, 0.15, 1.6)[None, :]
    heat = np.clip(demand + rng.normal(0, 0.03, (n_assets, n_hours)), 0.02, None)

    eff0 = rng.uniform(0.72, 0.80, (n_assets, 1))
    ramp = np.clip((t[None, :] - fault_start[:, None]) / (45 * HOURS_PER_DAY), 0, 1)
    eff = np.where(faulty[:, None], eff0 * (1 - 0.085 * ramp), np.repeat(eff0, n_hours, axis=1))
    ret = supply[None, :] - eff * (supply[None, :] - 28.0) + 4.5 * heat
    ret = ret + rng.normal(0, 0.55, (n_assets, n_hours))

    sup = np.tile(supply, (n_assets, 1))
    if shift_at is not None:
        sup[:, shift_at:] += shift_ambient               # supply-curve change
        heat[:, shift_at:] *= shift_load
    return dict(name="districtheating", X=np.stack([heat, sup / 80.0], -1),
                y=ret, faulty=faulty, fault_start=fault_start)


# ---------------------------------------------------------------- ensemble

class RidgeEnsemble:
    """Bootstrap ensemble of quadratic ridge models.

    mean  -> expected healthy behaviour
    std   -> epistemic disagreement, the abstention signal
    """
    def __init__(self, k=8, alpha=1e-3, seed=0):
        self.k, self.alpha, self.rng = k, alpha, np.random.default_rng(seed)
        self.W = []

    @staticmethod
    def _phi(X):
        a, b = X[..., 0], X[..., 1]
        return np.stack([np.ones_like(a), a, b, a * a, b * b, a * b], -1)

    def fit(self, X, y):
        P = self._phi(X).reshape(-1, 6)
        yy = y.reshape(-1)
        n = len(yy)
        for _ in range(self.k):
            idx = self.rng.integers(0, n, n)
            A, t = P[idx], yy[idx]
            self.W.append(np.linalg.solve(A.T @ A + self.alpha * n * np.eye(6), A.T @ t))
        return self

    def predict(self, X):
        P = self._phi(X)
        preds = np.stack([P @ w for w in self.W], 0)
        return preds.mean(0), preds.std(0)
