# Project Report: Testing the ABCD Rule for Explainable Melanoma Teaching

**Iteration 1 · Stages 1–2 complete · 2026-10-01**

> Educational project. Not for clinical use. Numbers are from the saved outputs in
> `reports/stage2/` and are reproducible with the commands in the README.

---

## 1. Purpose

Medical students learn the **ABCD rule** as a structured way to decide whether a
pigmented lesion is suspicious: Asymmetry, Border, Colour, and Differential structures
(dermoscopy) or Diameter (clinical). This project asks two questions:

1. **Does the rule actually work on modern, histopathology-labelled data?** We test the
   published Stolz weights and thresholds (TDS > 5.45 = highly suspicious) and compare
   them with what a model learns from the same four criteria.
2. **Can we show *how* an ABCD verdict is reached, case by case, in a way a student can
   follow and check?** The deliverable is a local web demo.

The success criteria (from `crisp/01_business_understanding.md`) include:
- sensitivity and specificity at the published thresholds, with confidence intervals,
  compared against a meta-analysis (pooled sensitivity 85 %, 95 % CI 73–93 %);
- interpretable models that beat chance and the fixed rule;
- automated measurements that agree with expert scores (κ ≥ 0.4);
- calibrated probabilities;
- under 5 seconds per case;
- clinician review of the explanations.

## 2. What the student sees (target design)

For one lesion the demo will show, in order:
1. the lesion outline the measurements came from;
2. the evidence behind each criterion (symmetry axes, abrupt border segments, colour
   swatches, detected structures or diameter);
3. the rule's arithmetic and threshold;
4. a calibrated probability from an interpretable model on the same features, showing
   where it agrees or disagrees with the textbook weights;
5. the most similar biopsy-confirmed cases;
6. a warning when the case falls in a known ABCD blind spot;
7. the true diagnosis.

A black-box model is used **only as a benchmark** for how much accuracy the
interpretable rule gives up. It is never presented as the explanation.

## 3. Data

| Dataset | What it is | Lesions / images | Melanomas | Used for |
|---|---|---|---|---|
| **ISIC 2024 SLICE-3D** | 15 × 15 mm crops from 3D total-body photography, with 34 machine-measured lesion features | 401,059 crops, 1,042 patients | 157 (133 patients) | Clinical ABCD(E) |
| **HAM10000** | Dermoscopic images + lesion masks | 10,015 images, 7,470 lesions | 614 lesions (all histology) | Dermoscopic TDS |
| **ISIC 2018 Task 2** | Expert masks of dermoscopic structures | 2,594 images | — | Validating the structure detectors (D) |
| **PH2** | Dermoscopy with expert A, C and structure scores | 200 | 40 | Validating our A and C measurements |

The derm7pt dataset was considered and dropped for this iteration. PH2's official
download link is dead, so it was taken from an unofficial mirror with the project
owner's approval. Its licence allows research and education only, with no
redistribution, so it stays local.

## 4. Key findings so far

### 4.1 Most "skin cancer" in ISIC 2024 is not melanoma
Of 393 malignant crops, **163 are basal-cell and 73 squamous-cell carcinomas**. Only
157 are melanoma. ABCD was designed for melanoma, so the primary target is now
**melanoma vs benign**, with *any malignancy* kept as a secondary analysis.
*Teaching point:* ABCD is not a general skin-cancer rule.

### 4.2 On clinical TBP images, Colour and Diameter carry the signal; Asymmetry and Border do not
Each ABCD-proxy feature on its own, melanoma vs benign (AUC; 0.5 = chance):

| Criterion | Feature | AUC |
|---|---|---|
| A | border asymmetry index | 0.62, **reversed**: melanomas measured *more* symmetric |
| B | border irregularity index | 0.53 |
| C | colour variation index | 0.80 |
| C | lesion-vs-skin contrast, relative to the patient's other lesions | 0.80 |
| D | lesion area / long diameter | 0.82 / 0.80 |

### 4.3 On dermoscopy with expert scoring, Asymmetry and Colour work well
On PH2, melanoma vs all nevi:
- **Expert asymmetry:** AUC 0.86. "Fully asymmetric" was scored in 33/40 melanomas,
  18/80 atypical nevi, and 1/80 common nevi.
- **Expert colour count:** AUC 0.87. The mean was 3.15 colours in melanoma vs about
  1.6–1.9 in nevi.

*Interpretation (to be tested in Stages 3–5):* the weak A and B signal in §4.2 most
likely reflects **image modality and measurement** (low-resolution body photos,
automated indices), not a failure of the clinical concept.

### 4.4 The "Diameter > 6 mm" rule misses many melanomas
In SLICE-3D, **39 % of melanomas are ≤ 6 mm**. The median long diameter is 7.0 mm and
the smallest is 1.1 mm. The > 6 mm rule therefore has sensitivity 0.61 and
specificity 0.90 for melanoma. *Teaching point:* small melanomas exist; do not
reassure on size alone.

### 4.5 Data pitfalls found (and how they are handled)
- **Label leakage.** `lesion_id` is filled in for 100 % of malignant crops but only
  5 % of benign ones. It, the diagnosis hierarchy, and the tumour-thickness fields are
  excluded.
- **Weak benign labels.** 99.8 % of benign SLICE-3D crops were never biopsied. We will
  also report results against the 561 histology-confirmed benign lesions, which is the
  clinically hard comparison.
- **Site shortcut.** Melanoma prevalence varies 8-fold across contributing hospitals.
  The hospital is not known at the bedside, so it is excluded as a feature and used
  for robustness checks.
- **Very few positives.** About 30 melanomas fit in a patient-grouped holdout, so every
  result will carry bootstrap CIs.
- **Prevalence mismatch.** HAM10000 has about 10 % melanoma vs 0.04 % in TBP screening.
  Predictive values will be re-weighted before being shown to students.
- **No border ground truth.** No public dataset scores border on Stolz's 0–8 scale. The
  automated B score can only be validated indirectly and by clinician review.

## 5. Published benchmarks we will compare against

| Study | n | Sensitivity / specificity |
|---|---|---|
| Nachbar 1994 (TDS > 5.45) | 172 | 92.8 / 90.3 |
| Argenziano 1998 | 342 | 85 / 66 |
| Ahnlide 2016 (bedside; 4.75 / 5.45) | 309 | 83 / 45 and 74 / 67 |
| **Harrington 2017 meta-analysis (TDS > 4.75)** | 8 studies | **85 (73–93) / 72 (65–78)** |

Inter-observer agreement on asymmetry is only fair to moderate (κ 0.35–0.49;
Rodríguez-Lomba 2022). So a κ of about 0.4 between our measurements and experts would
match human-level agreement.

## 6. Next steps

- **Stage 3, Data Preparation.**
  - Image-based A, B, C, D extraction on HAM10000 and PH2.
  - Patient-relative ("ugly duckling") features on SLICE-3D.
  - Investigate the reversed asymmetry index.
  - Leakage audit.
- **Stage 4, Modeling.**
  - Fixed Stolz rule vs an interpretable logistic/GAM on the ABCD features vs a
    black-box benchmark.
  - Patient-grouped holdout, fixed once.
- **Stage 5, Evaluation.**
  - Success criteria S1–S6.
  - Strong-label and leave-one-site-out analyses.
  - Clinician review and sign-off.
- **Stage 6, Deployment.** A local web demo of the case walk-through.

## 7. Process record
The full decision trail is kept in [`crisp/STATE.md`](../crisp/STATE.md): the
transition log and the human checkpoints with their dates and decisions.
