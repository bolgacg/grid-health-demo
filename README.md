# Alarm quality under regime change

A study of fault-detection architecture on two simulated energy systems, built around the PhD
position in Energy Informatics (Theme 2) at the SDU Center for Energy Informatics.

**Live page:** https://bolgacg.github.io/grid-health-demo/

The question: when the operating regime changes after a detector has been fitted, what happens to
the alarms an operator actually receives? The short answer is that detection rate and detection
delay both improve while the system becomes useless, and only precision reveals it.

- `index.html` + `app.js` — the interactive page. Vanilla JavaScript, no libraries, no build step.
  The simulation, the ensemble, the detection gates and the scoring all run in the browser.
- `study/` — the offline Python study behind the comparison table: two systems, five seeds,
  120 assets, 300 days. `python3 study/run_study.py` reproduces `study/results.json`.

Written by Bolgaç Gülen, August 2026.
