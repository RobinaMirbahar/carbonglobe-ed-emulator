# CarbonGlobe ED Emulator — Physics-Constrained Multi-Family Emulation & Rollout Modeling

**Author:** Robina Mirbahar (`@robinamirbahar`)  
**Competition:** [IEEE BigData Cup 2026: AI Emulation Challenge (CarbonGlobe)](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge)  
**Final Verified Submissions:** `sub_cal_v8_struct.csv` (Ref `58398629`, Public LB MSE `0.052`) & `sub_cal_v8_free_lam8.csv` (Ref `58398618`, Public LB MSE `0.052`)

---

## Overview

This repository accompanies my IEEE BigData Cup 2026 solution for 40-year Ecosystem Demography (ED) emulation of tree height (`H`) and aboveground biomass (`AGB`) across `6,171` global test grid sites, two climate scenarios (`ssp126`, `ssp585`), and `15` stand-age cohorts (`185,130` target pairs).

### Key Technical Contributions
1. **Physics-Hygiene Domain Correction:** Identifies and inverts a deterministic inverse-affine initial-state shift in test `y0` (`R² > 0.999`, `+14%` flux inflation, `+6.4%` `LAI`), removes corrupted test `CO2` channels (`8-32`) and deep soil channel `7`, and replaces raw shortwave levels with within-site years `0-4` anomalies.
2. **12-Family Multi-Scale Ensemble (`v4` + `v5`):** Combines `6` spatial-block families (`v4`, `469` checkpoints, spatial OOF `0.02849`, warming holdout `0.02444`, Public LB `0.079`) with `6` clean-physics families (`v5`, `481` checkpoints, stratified OOF `0.00956`).
3. **Structured Ridge CalibrationHead (`W_cal`):** Reconciles `v4` spatial stability with `v5` precision via a `(2, 15, 2, 55)` Ridge tensor anchored on `0.65 v5 + 0.35 v4` with cross-age smoothing `(0.12, 0.76, 0.12)`, achieving **`0.052` Public LB MSE** (`OOF 0.00536`) in deterministic single-pass inference.
4. **Coupled 40-Step Rollout Emulator (`v7 DeepEDRolloutGRU`):** `~824k`-parameter recurrent architecture with per-site RevIN, `CohortMixerBlock` cross-age self-attention (init, year 20, year 40), and Brandstetter rollout noise (`OOF 0.01371`, `warm90 0.01115`).

---

## Results Summary

| Suite / Model | Validation | OOF MSE | Shift / Warm | Public LB MSE |
| :--- | :--- | :---: | :---: | :---: |
| Persistence / `PCHIP40` | 5-fold Spatial Block | `0.29622` / `0.18103` | — | — |
| `v4` 6-Family Ensemble | `10° x 10°` Spatial Block | `0.02849` | `0.02444` | `0.079` (`58268845`) |
| `v5` Stage-1 Blend / `F12` Stack | 5-fold Stratified Site | `0.01075` / `0.00956` | — | — |
| `v7 DeepEDRolloutGRU` | 3-fold Stratified + `warm90` | `0.01371` | `0.01115` | `0.116` (`58398639`) |
| **CalibrationHead `v8` (`struct` & `free_lam8`)** | **Dual 5-fold Rand / Temp** | **`0.00536`** | **Base `0.067`** | **`0.052` (`58398629` / `58398618`)** |

---

## Deterministic Reproduction

Unzip the verified weights package (`03_Trained_Model_Weights_Robina_Mirbahar_CarbonGlobe_2026.zip`) and run:

```bash
pip install -r requirements.txt
python verify_package.py
python code/predict.py \
  --data_dir <path_to_public_test_data> \
  --ckpt_v4_dir checkpoints/v4 \
  --ckpt_v5_dir checkpoints/v5 \
  --cal_head checkpoints/cal/calibration_head_v8_struct.npz \
  --out_csv sub_cal_v8_struct.csv
```

Swap `--cal_head checkpoints/cal/calibration_head_v8_free_lam8.npz` to reproduce `sub_cal_v8_free_lam8.csv` (Ref `58398618`, `0.052` Public LB).
