# 04 — Modeling

**Project:** skin-cancer · **Iteration:** 1 · **Version:** 1.0 · **Date:** 2026-10-01

**Script:** `src/skin_cancer/model/train.py` (`uv run python -m skin_cancer.model.train`).

**Outputs:**
- `reports/stage4/run_log.csv`: every run
- `reports/stage4/modeling_summary.json`
- `reports/stage4/*_split.csv`: the fixed holdouts
- `reports/stage4/*_holdout_predictions.parquet`: saved for Stage 5, not scored here
- Fitted models in `~/data/skin-cancer/models/{dermoscopic,clinical}_models.joblib`

## 1. Evaluation data fixed first

| Module | Holdout | Training data | Model selection |
|---|---|---|---|
| Dermoscopic (HAM10000 primary cohort) | 25%, stratified, one image per lesion: **496 lesions, 154 melanomas** | 1,485 lesions, 460 melanomas | 5-fold grouped (lesion) stratified CV on train only |
| Clinical (ISIC 2024) | **25% of patients** (261), all their crops: **100,362 crops, 38 melanomas** | 781 patients; **all 119 melanomas + 20,000 random benign crops** (demo speed) | 5-fold patient-grouped CV on train only |
| External | **PH2**, 200 images (40 melanomas), never used for fitting | — | — |

PH2's A and C come from out-of-fold calibration.

**Holdout predictions are written but not scored in Stage 4.**

## 2. Candidates and why

These map to the Stage 1 tasks:
- **T2** (test the published rule): fixed rules with no fitting.
- **T3** (learn empirical ABCD weights): logistic regression, which is the most comprehensible probability model, with balanced class weights and L2 regularisation (C = 1).
- **T4** (cost of interpretability): gradient boosting as a black-box benchmark.

## 3. Run log

5-fold CV on train only. Copied from `run_log.csv`; pAUC80 ranges 0–0.2.

| Run | Module | Model | Features | CV AUC | CV pAUC80 |
|---|---|---|---|---|---|
| D0 | Dermoscopic | Baseline: constant | — | 0.500 | 0.020 |
| D1 | Dermoscopic | **Fixed Stolz TDS** (1.3 / 0.1 / 0.5 / 0.5) | A, B, C, D | 0.637 | 0.031 |
| D2 | Dermoscopic | Logistic regression | A, B, C, D | 0.687 | 0.044 |
| D3 | Dermoscopic | Logistic regression | 17 continuous ABCD measurements | 0.773 | 0.075 |
| **D3c** | Dermoscopic | Logistic regression | 17 ABCD + age, sex, site | **0.827** | 0.094 |
| D4 | Dermoscopic | **Benchmark**: gradient boosting | 30 features + context | 0.842 | 0.102 |
| C0 | Clinical | Baseline: constant | — | 0.500 | 0.020 |
| C1 | Clinical | **Fixed rule: diameter > 6 mm** | D | 0.742 | 0.046 |
| C1b | Clinical | **Fixed ABCD count** (A/B/C above train-benign P90, D > 6 mm) | A, B, C, D | 0.762 | 0.064 |
| **C2** | Clinical | Logistic regression | 7 ABCD core | **0.833** | 0.071 |
| C3 | Clinical | Logistic regression | ABCD + ugly-duckling z-scores | 0.830 | 0.075 |
| C3c | Clinical | Logistic regression | ABCD + ugly duckling + age, sex, site | 0.851 | 0.090 |
| C4 | Clinical | **Benchmark**: gradient boosting | All TBP features + DNN + context | 0.890 | 0.114 |

Hyperparameters:
- Logistic regression: C = 1, balanced class weights, `max_iter` 2000.
- Gradient boosting: 300 iterations, learning rate 0.05, 15 leaves, L2 = 1, balanced class weights.
- Seed 0.
- No hyperparameter search was run, so there is no hidden "best of many tries".

## 4. Selection rule and selected models
**Rule (fixed before running):** among the interpretable fitted models, take the highest CV AUC. If others are within 0.02, take the one with the fewest features. The black-box model is benchmark only and never selected.

- **Dermoscopic: D3c**, CV AUC 0.827. It beats D3 by 0.054, which is more than 0.02. Age, sex and site are allowed because they're known at the visit (01 §2).
- **Clinical: C2**, CV AUC 0.833. C3c (0.851) is within 0.02, so the simpler C2 wins.
- **Both beat their baselines** (0.50) **and their fixed published rules** (0.64 and 0.74–0.76).

## 5. Findings that matter for teaching
1. **The published Stolz cut-offs do not transfer to automated scoring.**
   - On train, only 11.1% of melanomas (and 3.2% of nevi) have automated TDS > 4.75.
   - The cut-off that gives the meta-analysis sensitivity of 85% on train is **TDS 1.68**.
   - The automated A, B, C and D are on a compressed scale: B is mostly 0, D is capped at 4, and C is noisy.
2. **Learned vs published weights** (D2, scaled so A = 1.3):

   | | A | B | C | D |
   |---|---|---|---|---|
   | Learned | 1.3 | 1.01 | **−0.16** | 0.29 |
   | Published | 1.3 | 0.1 | 0.5 | 0.5 |

   **Automated** colour count carries no signal. Expert colour count does (AUC 0.87 on PH2, 02 §3), so this reflects the colour detector, not the clinical concept. The rarely firing automated B gets a larger weight.
3. **Continuous measurements beat 0–2 / 0–8 / 1–6 sub-scores** (D3 0.773 vs D2 0.687). Discretising into Stolz points throws information away.
4. **Interpretability costs little.** The black-box benchmark gains only +0.015 AUC (dermoscopic) and +0.057 (clinical) over the selected interpretable models.
5. **The ugly-duckling features did not add AUC** in the clinical module (C3 0.830 vs C2 0.833).

## 6. Known weaknesses for Evaluation
- **C2 coefficients are collinear.** `norm_border` is a composite of jaggedness and asymmetry: it gets +8.4 while `area_perim_ratio` gets −5.2 and `symm_2axis` −4.3. Its *per-feature* explanations are therefore unstable, even though its predictions are fine. Stage 5 must decide whether to present the clinical module through the fixed rules and the univariate evidence instead.
- **Clinical training subsample:** 20,000 of about 300,000 benign crops. The predicted probabilities are not calibrated to the 0.04% base rate, so Stage 5 must recalibrate before showing probabilities.
- **Few clinical holdout positives (38 melanomas):** CIs will be wide.
- **HAM10000 has no patient ID:** two lesions from one patient can fall on either side of the split.
- **The structure CNN was trained on ISIC 2018**, whose images partly come from the same archive as HAM10000. There's no image overlap, but patient overlap can't be ruled out.
- **The pixel-CNN benchmark for T4 was skipped** (demo time). The gradient-boosting benchmark uses CNN structure probabilities, so the "cost of interpretability" estimate is a lower bound.

## 7. Iteration log
| Version | Date | Change | Why |
|---|---|---|---|
| 1.0 | 2026-10-01 | Initial modeling | Stage 4 first pass (demo scope) |

## Exit-gate self-check
- [x] **Holdout untouched.** Splits were saved before fitting. CV ran on train only. Holdout predictions are saved but unscored.
- [x] **Selected models beat the baseline on validation:** D3c 0.827 vs 0.50, C2 0.833 vs 0.50. Both also beat their fixed published rules.
- [x] **All selection steps logged:** `run_log.csv`, a selection rule fixed in advance, and no hyperparameter search.
