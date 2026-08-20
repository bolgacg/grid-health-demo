"""Study 3: cable lifetimes with censoring. A cohort of MV cable sections observed over a
25-year window with preventive replacements. Naive estimate vs Kaplan-Meier vs Weibull PH,
then two planning rules costed. (Shaker's survival review 2026; Mortensen & Shaker 2025.)"""
import numpy as np

def run(seed=7, n=400):
    rng = np.random.default_rng(seed)
    loading = rng.uniform(0.4, 1.0, n)                       # per-unit average loading
    scale = 46.0 * np.power(loading / 0.7, -0.9)             # heavier loading -> shorter life
    true_life = scale * np.power(-np.log(rng.uniform(size=n)), 1/3.4)   # Weibull(shape 3.4)
    install_age = rng.uniform(0, 30, n)                      # age at observation start
    window = 25.0
    # observed: fails inside window, or censored at window end / preventive replacement
    prevent_at = np.where(rng.uniform(size=n) < 0.3, rng.uniform(25, 45, n), np.inf)
    end_age = np.minimum(install_age + window, prevent_at)
    fails = true_life <= end_age
    obs_age = np.where(fails, true_life, end_age)
    # left-truncation ignored deliberately (kept simple); censoring is the point
    naive_mean = obs_age[fails].mean()                       # mean age of the ones seen dying
    # Kaplan-Meier
    order = np.argsort(obs_age)
    t_sorted, d_sorted = obs_age[order], fails[order]
    at_risk = np.arange(n, 0, -1)
    surv = np.cumprod(1 - d_sorted/at_risk)
    km_median = float(t_sorted[np.searchsorted(-surv, -0.5)])
    km_curve = [[float(t), float(s)] for t, s in zip(t_sorted[::6], surv[::6])]
    # Weibull fit on (obs_age, fails) by MLE grid (small, transparent)
    shapes = np.linspace(1.5, 6, 46); scales = np.linspace(30, 65, 71)
    ll_best, fit = -np.inf, (None, None)
    for sh in shapes:
        for sc in scales:
            z = obs_age/sc
            ll = np.sum(fails*(np.log(sh/sc) + (sh-1)*np.log(z+1e-12))) - np.sum(z**sh)
            if ll > ll_best: ll_best, fit = ll, (sh, sc)
    sh, sc = fit
    wb_median = sc * np.log(2)**(1/sh)
    # planning rules on the surviving population, next 10 years, budget = 60 replacements
    alive = ~fails
    age_now = end_age[alive]; load_now = loading[alive]; life_now = true_life[alive]
    horizon = 10.0; budget = 60
    will_fail = life_now <= age_now + horizon
    # rule A: replace oldest first
    order_a = np.argsort(-age_now)[:budget]
    # rule B: replace by fitted hazard (age and loading aware)
    hz = (sh/sc) * np.power(age_now/sc, sh-1) * np.power(load_now/0.7, 0.9*sh)
    order_b = np.argsort(-hz)[:budget]
    def outcome(chosen):
        prevented = int(will_fail[chosen].sum())
        wasted = int((~will_fail[chosen]).sum())
        misses = int(will_fail.sum() - prevented)
        return dict(prevented=prevented, wasted=wasted, in_service_failures=misses)
    return dict(
        n=n, n_failures=int(fails.sum()), n_censored=int((~fails).sum()),
        naive_mean=float(naive_mean), km_median=km_median, weibull=dict(shape=float(sh), scale=float(sc), median=float(wb_median)),
        km_curve=km_curve, rule_oldest=outcome(order_a), rule_hazard=outcome(order_b),
        will_fail_total=int(will_fail.sum()), budget=budget)
