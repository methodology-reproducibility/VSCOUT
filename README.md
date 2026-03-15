# Anonymous Repository for Peer Review

This repository contains code supporting a manuscript currently under peer review that proposes a hybrid representation-learning framework for retrospective (Phase I) anomaly detection in high-dimensional contaminated data.

The repository includes implementations used for the simulation study, ablation analysis, and contamination sensitivity experiments presented in the manuscript. All scripts are designed to reproduce the reported results and figures in a fully reproducible manner.

---

## Repository Contents

**Core Scripts**

* `ablation_framework.py`
  Runs the ablation study evaluating the contribution of individual components of the proposed framework.

* `contamination_sensitivity.py`
  Conducts experiments across increasing contamination levels.

* `plot_contamination_sensitivity.py`
  Generates the contamination-sensitivity figures reported in the manuscript.

---

## Reproducibility

All experiments were conducted using fixed random seeds. The scripts are structured so that the main simulation results and figures can be reproduced directly.

Typical workflow:

1. Run ablation study

```
python ablation_framework.py
```

2. Run contamination sensitivity experiments

```
python contamination_sensitivity.py
```

3. Generate figures

```
python plot_contamination_sensitivity.py
```

---

## Requirements

The code was developed in Python 3.10+. Required dependencies are listed in `requirements.txt`.

Install with:

```
pip install -r requirements.txt
```

---

## Notes

This repository is provided solely for anonymous peer review. Documentation and additional examples will be added after completion of the review process.
