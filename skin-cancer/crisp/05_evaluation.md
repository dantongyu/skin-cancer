# 05 — Evaluation

**Project:** skin-cancer · **Iteration:** 1 · **Version:** 1.0 · **Date:** 2026-10-01

All numbers come from `reports/stage5/evaluation.json` (script `src/skin_cancer/model/evaluate.py`). Each holdout was scored **once**. Intervals are stratified-bootstrap 95% CIs: 1,000 resamples for the dermoscopic holdout, 300 for the clinical one.

**What these results are conditional on:**
- The selection rule and the runs listed in 04 §3–4.
- Demo scope: the HAM10000 primary cohort, one image per lesion, and 20,000 benign ISIC 2024 training crops.
- A/C thresholds fitted on PH2 and D detectors trained on ISIC 2018 Task 2.

## 1. Validity: holdout results

### Dermoscopic holdout (HAM10000, 496 lesions, 154 melanomas)
| Model | AUC [95% CI] | pAUC80 | Specificity at 85% sensitivity |
|---|---|---|---|
| Baseline | 0.50 | 0.02 | — |
| D1 fixed Stolz TDS | 0.639 [0.584, 0.691] | 0.034 | 0.24 |
| D2 LR on A, B, C, D | 0.695 [0.648, 0.747] | 0.053 | 0.38 |
| D3 LR on continuous ABCD | 0.789 [0.750, 0.832] | 0.085 | 0.54 |
| **D3c LR + age, sex, site (selected)** | **0.847 [0.810, 0.879]** | **0.107** | **0.69** |
| D4 black-box benchmark | 0.829 [0.788, 0.867] | 0.091 | 0.61 |

Holdout results match CV (04 §3). There's no sign of overfitting: D3c's holdout AUC of 0.847 is close to its CV AUC of 0.827. **The interpretable model is at least as good as the black box.**

### External test (PH2, 200 images, 40 melanomas; different centre and device)
| Model | AUC [95% CI] |
|---|---|
| TDS (out-of-fold A/C) | 0.684 [0.589, 0.783] |
| D2 | 0.709 [0.609, 0.809] |
| **D3** (no context available in PH2) | **0.830 [0.739, 0.909]** |

The model generalises to a new centre.

### Clinical holdout (ISIC 2024, 261 patients, 100,362 crops, 38 melanomas, prevalence 0.038%)
| Model | AUC [95% CI] | pAUC80 |
|---|---|---|
| C1 diameter > 6 mm | 0.805 [0.726, 0.871] | 0.062 |
| **C1b ABCD count rule** | **0.841 [0.776, 0.894]** | **0.109** |
| C2 LR ABCD core (selected in Stage 4) | 0.804 [0.706, 0.896] | 0.033 |
| C3c LR + ugly duckling + context | 0.845 [0.756, 0.921] | 0.069 |
| C4 black-box benchmark | 0.933 [0.881, 0.972] | 0.149 |

**The Stage-4 selection (C2) does not hold up.** On the holdout, the simple fixed count rule matches C2 on AUC and beats it on pAUC80 (0.109 vs 0.033), and the CIs overlap. With 38 positives the evidence is weak, but there is no case for the fitted model. **Revised decision: the clinical module is presented through the fixed ABCD count rule and the diameter rule.** This is also more transparent, and avoids C2's collinear coefficients (04 §6).

- **Strongly-labelled subset** (melanoma vs biopsied benign; 178 crops, 38 melanomas): C2 AUC 0.64, diameter alone 0.73. ABCD discriminates far less well against clinically suspicious benign lesions.
- **By institution:** C2 AUC is 0.98 at Barcelona (9 melanomas) and 0.75 at MSKCC (21 melanomas). Site heterogeneity is likely.

## 2. Success criteria (01 §4b)

| ID | Criterion | Result | Verdict |
|---|---|---|---|
| **S1** | Published TDS cut-offs reproduce published sensitivity/specificity | **Automated TDS > 4.75:** sens **0.084** [0.046, 0.130], spec 0.988 (holdout). PH2: sens 0.225. Published pooled: **0.85** [0.73, 0.93] / 0.72 [0.65, 0.78]. **Recalibrated TDS ≥ 1.68** (set on train): sens 0.877, but spec only 0.234 | **Not confirmed for *automated* TDS.** The weights and cut-offs assume expert-scored criteria. Our automated sub-scores are compressed (B rarely fires, D is capped at 4, C is noisy), so no single cut-off gives published-level sensitivity and specificity together. This does **not** refute the rule for expert scorers: expert A and C on PH2 discriminate at AUC 0.86 / 0.87 (02 §3) |
| **S2** | Interpretable ABCD model beats chance and the fixed rule | Dermoscopic D3c 0.847 vs TDS 0.639 vs 0.5 | **Met** (dermoscopic). Clinical: the fixed count rule is retained (see §1) |
| **S3** | Cost of interpretability reported | Dermoscopic: black box −0.018 vs D3c (no cost). Clinical: black box +0.09 vs the count rule | **Reported** |
| **S4** | Feature extraction agrees with experts, κ ≥ 0.4 | A: κ 0.398. C (count): κ 0.10–0.23. Dots/globules: κ 0.42. Streaks: κ 0.22 (03 §2) | **Partly met.** A and dots/globules sit at the threshold, within the human inter-observer range of 0.35–0.49. **C and streaks fail.** B has no ground truth |
| **S5** | ECE ≤ 0.05 after base-rate correction | D3c: raw 0.101 → **0.037** after prior correction | **Met.** Probabilities must be shown with the correction, and only for the referral setting |
| **S6** | ≤ 5 s per case | **0.67 s** (extraction + CNN + model, M2) | **Met** |

## 3. Business value

The aim is to teach students to triage. The expected-value lens from 01 §4a, through false-alarm burden:

**Dermoscopic referral setting** (melanoma prevalence 8.2%, the HAM10000 lesion rate):
- Recalibrated TDS (sens 0.88): **NNE 10.8** excisions per melanoma.
- D3c at 85% sensitivity: **NNE 5.1**.

Both are within the range reported for dermatologists, who are often cited at about 5–15 excisions per melanoma (a literature range not checked in this project).

**Clinical TBP screening setting** (prevalence 0.038%):
- Diameter > 6 mm flags **10,117 per 100,000 crops**: sens 0.71, spec 0.90, **NNE 376**.
- The ABCD count ≥ 2 rule: sens 0.61, **NNE 560**.

**A single ABCD rule applied to every lesion on a body is not a usable triage tool.** This is a key teaching message: ABCD is for lesions that already attract attention, not for screening every mole.

**Blind spots found** (T6):
- **Symmetric melanomas.** The 23 holdout melanomas D3c misses have mean A 0.09, vs 0.66 for detected ones.
- **Tiny melanomas.** All 4 holdout melanomas ≤ 2.74 mm rank in the bottom 2–18% of C2 scores, so they would be missed. The demo must show the warning: *"ABCD can't exclude small or symmetric melanoma."*

## 4. Practical and external constraints
- **Licences.** ISIC and HAM10000 are CC-BY-NC, and PH2 is no-redistribution. A **local** demo is fine. PH2 images must not be published.
- **Not a medical device.** A disclaimer is required on every page.
- **Probabilities** are calibrated only for the referral (dermoscopic) setting. The clinical module shows rule outputs and rank percentiles, not probabilities.
- **Colour (C) and streak detection are unreliable.** The demo labels them "automated, low agreement with experts" and shows the evidence so students can judge.

## 5. Stakeholder review and sign-off
- **Owner sign-off:** see the record below. This is a *local educational demo* only.
- **Clinical content review** by a dermatologist: **outstanding**. It is required before any use with real students (01 §5). The demo is labelled "prototype — not clinically reviewed".
- **Requested exceptions:** none yet.

**Sign-off record**

| Date | Who | Decision | Conditions |
|---|---|---|---|
| 2026-10-01 | Project owner (dantongyu) | **Approved: local educational demo** (scope clarified the same day: personal demo for self + a few collaborators, no student use) | Labelled 'prototype, not clinically reviewed'; local only; PH2 images not published; no real-patient use |

## 6. Testbed and in-vivo
- **Testbed:** the local web demo on the held-out cases, which never influenced training, is the testbed. The PH2 external result (AUC 0.83) is the closest thing to a production-like test.
- **In-vivo:** a randomised study of student learning (pre/post quiz) is the natural design. It needs a curriculum owner and an ethics decision, so it is **not run** in iteration 1.

## 7. Monitoring plan (for the demo)
- Log every case shown: the case ID, the student's A/B/C/D scores, the model's sub-scores, and the time taken. Store this locally with no personal data.
- **Alert if:**
  - an uploaded image fails segmentation (mask < 200 px);
  - skin normalisation shifts Lab by more than 25 units (unusual device or lighting);
  - CNN structure probabilities sit at 0 or 1 for more than 95% of a batch (input drift).
- **Re-evaluate** whenever detectors or thresholds change, using the same frozen holdouts. Holdouts are scored once per model version.

## 8. Decision
- **Dermoscopic module:** deploy as a **local educational demo**. It shows the transparent Stolz arithmetic (with the published cut-offs *and* the finding that they don't transfer to automated scoring), D3c's per-feature contributions, similar holdout cases, and blind-spot warnings.
- **Clinical module:** deploy as a **teaching panel**. It shows the fixed ABCD count and > 6 mm rules on TBP data, with the false-alarm (NNE) lesson. No fitted model is shown.
- **Iteration 2 should address:**
  - better C and B measurement, or student-entered scores;
  - dots vs globules as separate structures;
  - patient-level splits for HAM10000;
  - the full ISIC 2024 training set;
  - a dermatologist review.

## Exit gate → Stage 6
- [x] The holdout result holds up (dermoscopic). The clinical selection was revised to the fixed rule on the holdout evidence, with no re-tuning.
- [x] Business criteria are met for an educational demo, with the screening false-alarm limitation documented as a teaching point.
- [x] **Stakeholder sign-off:** project owner approved a local educational demo on 2026-10-01. Clinical review is still outstanding and is a condition for any use with real students.
- [x] Monitoring plan exists (§7).
