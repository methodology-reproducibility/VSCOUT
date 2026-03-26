# Anonymous Repository for Peer Review

This repository contains code supporting a manuscript currently under peer review that proposes a hybrid representation-learning framework for retrospective (Phase I) anomaly detection in high-dimensional contaminated data.

The repository includes implementations used for the simulation study, ablation analysis, and contamination sensitivity experiments presented in the manuscript. All scripts are designed to reproduce the reported results and figures in a fully reproducible manner.

---

## Repository Contents

**Core Scripts**

* `ablation_framework.py`  
  Runs the ablation experiments used to evaluate the contribution of ARD, changepoint filtering, ensemble detection, refinement, and the full VSCOUT procedure.

* `consensus_threshold_sensitivity.py`  
  Conducts the sensitivity analysis for alternative final decision rules, comparing 1-of-4, 2-of-4, and 3-of-4 consensus thresholds.

* `hyperparameter_sensitivity.py`  
  Runs the hyperparameter sensitivity experiments for latent size, KL threshold, PELT penalty, significance level, ensemble rule, and hidden-layer width.

---

**Experiment Notebooks**

* `Experiments (Synthetic Shifts).ipynb`  
  Contains the main simulation experiments for retrospective monitoring under synthetic shift scenarios.

* `Experiments (Benchmark Comparison).ipynb`  
  Provides benchmark comparisons against competing methods across the experimental settings reported in the manuscript.

* `Ablation Study (Sensitivity_to_contamination_Figure).ipynb`  
  Generates the contamination-sensitivity experiments and corresponding figure showing how performance changes as contamination increases.

* `Ablation Study (Ablation Study Table).py`  
  Produces the ablation-study summary table reported in the manuscript.

* `Semiconductor.ipynb`  
  Contains the real-data semiconductor case study and associated analyses.

---

**Reproducibility / Figure Generation**

* `Code to Reproduce Figures.ipynb`  
  Reproduces the main figures reported in the manuscript.

* `Code to Reproduce Control Charts.ipynb`  
  Reproduces the control-chart visualizations used in the paper.

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
