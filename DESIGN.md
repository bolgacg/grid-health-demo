# Design brief — rebuild of the 4159 demo (2026-08-20)

Bo's critique of v1: too busy, not interesting enough, no visual representation of the grid,
conclusions hard to interpret. Requirements: visually represent an electricity grid; make the
resultant data easily interpretable; ground every main concept in the professors' papers and in
the papers they themselves cite heavily; impress Shaker and Jørgensen specifically.

## What the research says about these two readers

**Shaker** (from 23 of his papers on disk): his through-line is *distribution-grid maintenance
under uncertainty*. Recurring strands: fault detection trained only on healthy data with
MC-dropout uncertainty in the ensemble vote (2026, Energy & Buildings); overload alarms on
distribution transformers, later made probabilistic with the epistemic/aleatoric split (2024 →
2026); cable replacement via Neural Weibull proportional hazards under data deficiency
(Mortensen & Shaker 2025); survival models for RUL where censoring is the whole problem (2026
review); consequence-aware alarm ranking because crews are finite (2025 TSG); XAI because
operators cannot act on black boxes (2025 review, 89 cites). His reference lists over-index on:
LSTM, autoencoder residuals, Weibull/Cox/Kaplan-Meier, SHAP/LIME, Temporal Fusion Transformer,
MC dropout (Gal & Ghahramani). His own most-cited co-authors: Mirshekali & Dashti (fault
location), Mortensen/Skydt/Bang (reliability & RUL).

**Jørgensen** (from 30 papers): systems and architecture, not detectors. Deployment feasibility
as a *sequence of non-compensable gates* (Ma, Cong & Jørgensen 2026, Energies): a strategy that
fails an early gate cannot buy its way back with a good score later. Control-room situation
awareness as a layered architecture (Infostructure: Jørgensen & Ma 2026, Energies), built on
Endsley's three levels: perceive → comprehend → project. Digital-twin frameworks for
distribution grids with DER (2024, Energies). MLOps lifecycle capability mapping (2026).
Co-authors: Zheng Grace Ma on nearly everything structural, plus Værbak and Howard.

## The concept: a small control room, not a dashboard

One page that behaves like a miniature DSO control room for a simulated 10 kV distribution
network. Named: **"Twenty-four transformers, four cables, one operator"** (working title).

**Centre stage — the grid, drawn.** An SVG one-line diagram: one 60/10 kV primary substation,
four radial feeders, 24 distribution transformers, cable segments as edges. Time plays over ~300
simulated days. Health states render on the diagram itself: assets glow calm teal when healthy,
ember-amber as a fault develops, red when failed; tickets appear as pins on the map. A regime
change (heat wave + EV uptake) visibly loads the whole network mid-run. The reader *watches* a
slow transformer fault emerge, sees who catches it and when.

**Right panel — the selected asset's story, in Endsley's three levels** (the Infostructure
structure, deliberately): L1 PERCEIVE = its raw sensor traces; L2 COMPREHEND = detector state:
forecast residual, ensemble vote, MC-dropout uncertainty band, CUSUM; L3 PROJECT = its Weibull
survival curve and inspect-by date. The panel is labelled with the three levels by name.

**Top strip — verdicts, not tables.** Three big tiles, each with its one-sentence English
meaning ("31 of 38 tickets pointed at genuinely sick assets" beats a precision column). The
architecture comparison becomes ONE paired-bar visual: warning-days gained vs false call-outs
per month. The Monday Morning List: a ranked work-order queue — asset, plain-words reason
(top drivers), confidence, act-by date — which is Shaker's 2023 "anomaly indexing for
maintenance decision support" made literal.

**One master control:** the alarm budget slider ("how many false call-outs per month will your
crew accept?"), which moves every threshold at once and re-scores the run. Scenario toggle:
stationary vs regime change. That is all the interactivity: click asset, drag budget, toggle
world, scrub time.

**The honest failure, kept:** under regime change the naive detector floods the queue with
confident nonsense; the uncertainty-aware gate abstains and says so on screen. A visible banner:
"the model knows it is confused" with the cost counter of both failure modes.

**Studies (offline Python, results shipped as JSON, seeded, reproducible):**
1. *The cost of certainty* — five detector architectures (threshold → learned-forecaster
   residual → multi-method vote → +MC-dropout-style uncertainty gate → +CUSUM persistence),
   scored in operator units: warning days vs false call-outs.
2. *The day the world changed* — same architectures across the regime change; the gate + the
   recalibration trade.
3. *When do you replace the cable?* — lifetimes with censoring: the naive estimate vs
   Kaplan-Meier vs Weibull hazard; what each planning rule costs in premature replacements and
   in-service failures.

**Academic provenance section — a mapped lineage, not a bibliography.** For each mechanism on
the page, one card: mechanism → the centre's paper (author, year, venue) → the foundational work
it stands on. Shaker's papers tagged as his, Jørgensen's as his, the shared citation canon
(Gal & Ghahramani; Page; Kaplan & Meier; Cox; Ishwaran; Katzman; Lundberg & Lee; Ribeiro;
Endsley; Gama) as the third column. Includes group co-authors by name (Mortensen, Mirshekali,
Ma, Howard) — the signal that Bo read the group, not one landing page.

## Interpretability rules (the fix for v1)

- Every number on screen is paired with a sentence saying what it means for a crew or a budget.
- Nothing is shown in units a maintenance planner does not use: days of warning, call-outs per
  month, replacements per year. No AUC, no abstract precision tables above the fold.
- One view at a time: map first; studies each get one visual and one verdict line; academic
  section separate, at the end.
- Dark control-room aesthetic — the page should look like the subject.

## Deliberate limits, stated on the page

Simulated network, simplified physics, ridge ensemble standing in for the deep models (the
uncertainty logic is what is being demonstrated, not the architecture); district heating not
modelled but the same methods are the centre's DH agenda (say so, cite it).
