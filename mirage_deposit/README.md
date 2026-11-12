# Code and derived data for *Trustworthy Optical Sensor Arrays for Chemical-Threat Identification*

Supporting code and derived data for the invited Review submitted to *Sensors and Actuators B: Chemical* (manuscript SNB-D-26-08464). This archive supports the two reproducible worked examples and the DOI-verified literature audit (evidence map) in the Review. It contains **derived descriptors and analysis code only**; raw authentic-agent images are subject to release restrictions and are not included.

## Contents

```
evidence_map.csv / evidence_map.json   Study-level literature audit (machine-readable),
                                       with batch/temporal-drift and device/operator/site-transfer
                                       scored as two separate axes, plus per-cell basis notes.
reference_crossref_check.json          Crossref verification log for every cited DOI.
code/
  optical_worked_example.py            Acquisition-provenance test on the colorimetric delta-E array.
  gas_drift_sizematched_control.py     Gas-sensor temporal-drift example + size-matched control.
data/
  optical_deltaE/d{1,2,3}.{normal,uv}.csv   Derived 29-dye delta-E feature vectors
                                       (one vector per agent-concentration-illumination condition).
results/
  optical_results.json                 Per-seed within/cross/LOCO accuracies, drop, bootstrap CI,
                                       permutation tests (canonical run).
  perm_test_summary.json               Label-permutation summary (within vs cross illumination).
  gas_drift_results.json               Pooled-CV / batch-1-transfer / size-matched control decomposition.
```

## Optical worked example (acquisition-provenance test)

Each sample is one 29-dye **ΔE** response vector for a single (panel, agent, concentration, illumination) condition; the classification target is the chemical-threat agent (K = 17 across three panels d1/d2/d3). This is the **experimental/resampling unit**: there are no independent physical replicates. Concentration-0 blanks are all-zero vectors that cannot be assigned to an agent class and are **excluded** from the closed-set classification. The two illumination sets (ambient "normal", ultraviolet "uv") image the **same physical exposures**, so the cross-illumination comparison is paired within exposure.

Protocols: within-illumination stratified 4-fold CV (10 seeds); leave-one-concentration-out; cross-illumination transfer (train one illumination, test the other). Random forest (300 trees), features standardized on the training partition only. Significance by label-permutation test (B = 300) on a single held-out split; no Cohen's d is reported because a standardized effect size is not well defined without physical replicates. The bootstrap CI on the transfer drop resamples the ΔE condition vector (2000 resamples).

Headline results (see `results/optical_results.json`): within-illumination ≈ 0.56 (0.52 ambient / 0.60 ultraviolet; chance 0.059) → cross-illumination ≈ 0.08 (both directions; permutation p ≈ 0.4, indistinguishable from chance); leave-one-concentration-out ≈ 0.67 (does not collapse). Reproduce: `python3 code/optical_worked_example.py` (reads `data/optical_deltaE/`).

## Gas-sensor drift example and size-matched control

Public UCI Gas Sensor Array Drift dataset (Vergara et al. 2012; 10 batches, 6 classes, 128 features). **Not redistributed here** — download `batch1.dat … batch10.dat` from the UCI Machine Learning Repository (dataset 224) into a `Dataset/` folder and set the path at the top of the script. The script computes: pooled random 5-fold CV (A); batch-1 → later-batch transfer (C); and a **size-matched control** (B) that trains on an equally small sample drawn at random from the pooled data, isolating the training-set-size effect from temporal/compositional shift. Result (`results/gas_drift_results.json`): A = 0.99, B = 0.93, C = 0.49; of the ≈ 0.50 gap, training-set size explains ≈ 0.07 and temporal + class-composition shift ≈ 0.44.

## Evidence map

`evidence_map.*` record the full study-level audit. The main-text counts are computed over the 21 primary optical sensor-array studies (`primary_optical_array = 1`): known-class 21/21; authentic-agent 4/21; grouped/independent-unit 0 explicit (12 partial); batch/temporal drift 0/21; device/operator/site transfer 0 explicit (6 partial); open-set/OOD 0/21. `basis_notes` records why each drift/device cell was scored as it was.

## License

Code: MIT. Derived data and evidence map: CC-BY-4.0. Please cite the Review (SNB-D-26-08464) and this archive.
