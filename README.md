# When does a fault prediction deserve to be believed?

An interactive study of predictive-maintenance detector architectures on a simulated 10 kV
distribution network, built from the published methods of the SDU Center for Energy Informatics. Every mechanism is taken
from the centre's own publications; the provenance section on the page maps each one to its
source and to the foundational work it stands on.

Live: https://bolgacg.github.io/grid-health-demo/

## What it shows

- A one-line diagram of 24 distribution transformers on four feeders, played over 300 days.
  Three cooling faults develop, one slowly enough to hide.
- Five detector architectures, each adding one idea from the centre's papers: static limit,
  learned-forecaster residual, multi-method vote, uncertainty-weighted vote, and a full
  sequential gate (CUSUM persistence). A sixth arm retrains weekly.
- The finding: under a regime change, weekly retraining produces the cleanest work-order
  queue on the board and is the only architecture that misses the slow fault. Retraining
  absorbs slow decay into the definition of normal, and every redeploy restarts the
  evidence accumulator.
- A censoring-aware cable-replacement study: naive lifetime estimates vs Kaplan-Meier vs a
  fitted Weibull hazard, costed in prevented failures at a fixed budget.

## Reproduce

```
python3 study2/run_study.py   # seeds 0-4, ~1 s, numpy only
```

Outputs `study2/results.json` (verdicts) and `study2/timeline.json` (the browser payload,
bundled as `data.js`). The page is static: no build step, no dependencies.

Simulated data throughout; the page says so. Bolgaç Gülen, August 2026.
