# 03 — Data Preparation

**Project:** skin-cancer · **Iteration:** 1 · **Version:** 1.0 · **Date:** 2026-10-01
**Inputs:** `01_business_understanding.md` v1.2, `02_data_understanding.md` v1.1

Every number here comes from `reports/stage3/`.

**Scripts**, all re-runnable:

| Script | Command | What it does |
|---|---|---|
| `src/skin_cancer/prep/isic2024.py` | `uv run python -m skin_cancer.prep.isic2024` | Clinical module |
| `src/skin_cancer/prep/abcd_image.py` | (library) | Dermoscopic A/B/C/D extractor, with evidence for the demo |
| `src/skin_cancer/prep/dermoscopy.py` | `… prep.dermoscopy ph2 task2 ham ph2val finalize` | Calibration, extraction, final scores |
| `src/skin_cancer/prep/structure_cnn.py` | `… prep.structure_cnn cache train infer ph2val` | Learned structure detectors for D |

**Outputs** go to `~/data/skin-cancer/prepared/`, outside the repo because they are derived from licensed data:
- `isic2024_features.parquet`
- `ham10000_model.parquet`
- `ph2_features.parquet`

The structure-detector weights are in `~/data/skin-cancer/models/`.

**Demo scope (user instruction, 2026-10-01).** The dermoscopic module uses the HAM10000 primary cohort with one image per lesion (n = 1,981), not all 10,015 images.

---

## 1. Unit of analysis, prediction time, target

| Module | Unit | Prediction time | Primary target | Secondary target | Grouping key |
|---|---|---|---|---|---|
| Clinical (ISIC 2024) | One TBP lesion crop | At the total-body photo | `y_melanoma`: melanoma (incl. in situ) = 1, benign = 0, BCC/SCC = NaN | `y_malignant` (incl. BCC/SCC) | `patient_id` |
| Dermoscopic (HAM10000) | One lesion (first image) | At the dermoscopy visit | `y_melanoma`: mel = 1, nv = 0, **histology-confirmed only** | — | `lesion_id` |
| External (PH2) | One image | At the visit | melanoma (2) vs nevi (0, 1), clinical diagnosis | — | — |

### Row counts

**ISIC 2024** (`isic2024_row_counts.csv`)

| Step | Rows | Malignant |
|---|---|---|
| Loaded and joined | 401,059 | 393 |
| Drop Indeterminate | 400,945 | 393 |
| Usable for the melanoma target | 400,709 | 157 melanomas |

**HAM10000** (`ham10000_row_counts.csv`)

| Step | Images | Melanomas |
|---|---|---|
| All images | 10,015 | 1,113 |
| Histology-confirmed mel + nv, one image per lesion | 1,981 | 614 |

The second HAM10000 step leaves 1,367 nevi. All 1,981 rows extracted with 0 errors, and all 10,015 images extracted without error in the full run (`ham10000_extraction_summary.json`).

**PH2:** 200 images, 40 melanomas.

## 2. Features

### 2a. Clinical module (ISIC 2024)
Full table: `isic2024_data_dictionary.csv`. Groups: `isic2024_feature_groups.json`.

| Criterion | Features | Notes |
|---|---|---|
| A | `tbp_lv_symm_2axis`, `tbp_lv_radial_color_std_max` | Lesion Visualizer, 0–10 scales |
| B | `tbp_lv_norm_border`, `tbp_lv_area_perim_ratio` | |
| C | `tbp_lv_norm_color`, `tbp_lv_deltaLBnorm` | |
| D | `clin_size_long_diam_mm`; `D_gt6mm` (published > 6 mm rule) | Calibrated mm |
| "Ugly duckling" | `*_pz`: z-score of each core feature among **the same patient's lesions** | Computed before any row filtering, without labels |
| Context | `age_approx`, `sex`, `anatom_site_general`, plus missing-value indicators | Imputation is fitted inside CV folds (Stage 4), never on the holdout |
| Benchmark only | All other `tbp_lv_*`, the two DNN confidences, `n_lesions_patient`, tile type, detailed location | Never shown as an explanation |

### 2b. Dermoscopic module (HAM10000, PH2)
Full table: `dermoscopy_data_dictionary.csv`. Thresholds and their provenance: `dermoscopy_thresholds.json`.

**Preprocessing**
- Resize to 512 px on the longest side.
- Clean the lesion mask: keep the largest component and fill holes.
- Remove hair (DullRazor-style).
- **Skin-referenced Lab normalisation**: the perilesional skin is mapped to the PH2 median skin colour (74.1, 13.8, 18.4) (`ph2_lab_reference.json`). The usual shades-of-gray colour constancy was rejected because it shifted browns towards blue-gray (02 → 03 finding).

| Criterion | Method | Calibration and validation |
|---|---|---|
| **A** (0–2) | Search 18 rotations for the least-asymmetric orthogonal axis pair. An axis counts as asymmetric if contour XOR > 0.225 or half-to-half colour ΔE > 10.0 | Thresholds fitted on PH2 expert A with 5-fold OOF. **OOF quadratic κ = 0.398** (`ph2_abcd_validation.json`) |
| **B** (0–8) | In each 45° sector, the pigment ramp width divided by the equivalent diameter. A sector is abrupt if the ratio is < 0.03 | **No public ground truth**. Heuristic. On PH2, B = 0 in 164 of 200 images; mean B by diagnosis is 0.10 / 0.16 / 0.62 for common nevus / atypical nevus / melanoma |
| **C** (1–6) | Nearest Lab reference colour among white, red, light brown, dark brown, blue-gray, black, plus a skin pseudo-class. A colour is present if its pixel fraction ≥ a per-colour threshold | Per-colour thresholds fitted on PH2 with 5-fold OOF. Count: **quadratic κ 0.099, exact agreement 0.46, within ±1 0.885**. A single global threshold gives κ 0.226 but only 0.39 exact agreement; it is saved but not used. Per-colour κ: light brown 0.53, dark brown 0.37, black 0.36, blue-gray 0.26, **red −0.01, white 0.07 (unreliable)** |
| **D** (1–4) | **Learned detectors** (ResNet18, concept bottleneck) for pigment network, dots/globules and streaks, plus a heuristic structureless area. Task 2 labels dots and globules **jointly**, so they count as one structure, making D range 1–4 instead of Stolz's 1–5 | See §2c |

### 2c. Structure detectors (a measurement model, kept in Stage 3 by user decision)

**What it is and how it was trained**
- ResNet18, ImageNet-pretrained.
- Trained on lesion crops from **ISIC 2018 Task 2**: 2,594 images, split 80/20 (2,075 train / 519 validation). 0 images overlap HAM10000.
- 8 epochs on the M2 GPU (MPS). The best epoch by mean validation AUC was the final one (epoch 7).
- Targets are structure presence only. It **never sees a melanoma label or HAM10000 for training**.
- Thresholds were chosen by Youden's J on the validation split.

**Validation AUC, ISIC 2018 Task 2 hold-out** (`structure_cnn_training.json`)

| Structure | Hand-built detector | CNN | CNN sensitivity / specificity |
|---|---|---|---|
| Pigment network | 0.68 | **0.835** | 0.78 / 0.76 |
| Dots/globules | 0.56 | **0.847** | 0.75 / 0.84 |
| Streaks | **0.33** (worse than chance) | **0.907** (20 positives) | 0.90 / 0.81 |

Hand-built figures are from `task2_d_calibration.json`.

**External PH2 validation** (`structure_cnn_ph2_validation.json`)

| Structure | AUC | κ |
|---|---|---|
| Dots/globules | 0.779 | **0.417** |
| Streaks | 0.760 | 0.219 |

PH2 marks a network in every image, so the network detector can't be tested there. It fired on 64% of PH2 images.

**Resulting score distributions** (`ham10000_score_distributions.json`, `ph2_d_validation.json`)

| | HAM10000 melanoma | HAM10000 nevus | PH2 common nevus | PH2 atypical nevus | PH2 melanoma |
|---|---|---|---|---|---|
| Mean TDS | **3.04** | **2.47** | 2.64 | 3.00 | **3.63** |

PH2 scores use out-of-fold A and C. Automated TDS is shifted well below the published scale (thresholds 4.75 / 5.45), mainly because B rarely fires and D is capped at 4. **Stage 4 must test the published thresholds and a recalibrated threshold separately.**

## 3. Leakage audit

| Feature or field | Available at decision time? | Verdict |
|---|---|---|
| ABCD Lesion Visualizer features, diameter | Y: produced by the TBP scan | KEEP |
| Patient-relative `_pz` | Y: TBP captures all of the patient's lesions in one session, and no labels are used | KEEP. Splits must be grouped by patient, which they are |
| Age, sex, site | Y: visible or askable | KEEP (context) |
| DNN confidences, other `tbp_lv_*` | Y, but black-box | Benchmark only |
| `lesion_id` (ISIC 2024) | Recorded *because* the lesion was of interest: present on 100% of malignant vs 5.4% of benign crops | **DROP (leak)** |
| `iddx_*`, `mel_thick_mm`, `mel_mitotic_index` | N: histopathology | **DROP (leak)** |
| `attribution`, `copyright_license` | Not a clinical variable; prevalence varies 8-fold by institution | **DROP** (shortcut). Kept for leave-one-site-out checks |
| `strong_label` | N: equivalent to "was biopsied" | Evaluation subsetting only |
| HAM10000 `dx_type` | N: histo means the lesion was excised | **DROP** as a feature. Used only to *define* the cohort |
| HAM10000 `dataset` | Device/collection: MoleMax is 3,720 nv vs 24 mel | **DROP**. Neutralised by the histology-only cohort (MoleMax histo = 48 images) |
| Image-derived A/B/C/D, CNN structure probabilities | Y: computed from the image at the visit | KEEP. Thresholds were fitted on PH2 and Task 2, **never on the Stage 4 holdout** |
| Structure CNN | Trained on Task 2 only, 0 overlap with HAM10000 | KEEP (frozen) |

**Suspiciously strong predictors:** none among the kept features. The strongest single ISIC 2024 feature has a melanoma AUC of about 0.82 (02 §3), which is plausible.

## 4. Investigations recorded for Evaluation
1. **The reversed TBP asymmetry index is not a size artefact.**
   - Spearman ρ with diameter = −0.029 (`isic2024_size_correlation.json`).
   - In the largest-size quintile (122 of the 157 melanomas), AUC is 0.368 for asymmetry and 0.404 for border, while colour is 0.792 (`isic2024_asymmetry_size_check.csv`).
   - The automated A and B indices on TBP crops do not behave as the textbook expects. This is a teaching point.
2. **21 melanomas ≤ 2.74 mm** in the smallest quintile, where colour AUC is reversed (0.26). This suggests crop or segmentation issues. Flagged for Stage 5 error analysis.
3. **Expert-scored dermoscopic A and C discriminate well on PH2** (AUC 0.86 / 0.87, 02 §3). Automated A matches experts at κ 0.40, but automated C is weak. Measurement error, not the clinical concept, is the bottleneck.

## 5. Iteration log
| Version | Date | Change | Why |
|---|---|---|---|
| 1.0 | 2026-10-01 | Initial preparation | Stage 3 first pass |
| — | 2026-10-01 | Shades-of-gray replaced by skin-referenced Lab | Browns were classified as blue-gray |
| — | 2026-10-01 | Heuristic D replaced by learned detectors | Heuristics had AUC 0.33–0.68 on Task 2 (user chose the learned option) |
| — | 2026-10-01 | HAM10000 limited to one image per lesion, primary cohort (n = 1,981) | User: demo scope, speed |

## Exit-gate self-check
- [x] **Every feature passes the decision-time availability check** (§3). Leaks and shortcuts are dropped. Evaluation-only fields are flagged.
- [x] **Preparation is fully scripted and re-runnable** (four scripts listed above). Raw data is never edited by hand.
- [x] **Data dictionary complete**: `isic2024_data_dictionary.csv` and `dermoscopy_data_dictionary.csv`, plus the excluded-field lists.
- Cross-stage invariants:
  - Features match the 01 use scenario (image plus age, sex, site at the visit).
  - No holdout exists yet, and nothing has been fitted on Stage 4 data.
  - No licensed data is in the repo.
