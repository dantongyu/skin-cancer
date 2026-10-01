# 01 — Business Understanding

**Project:** skin-cancer · **Iteration:** 1 · **Version:** 1.0 · **Date:** 2026-10-01

> **Scope disclaimer (applies to every artifact this project produces):** this is an
> *educational* system for medical students. It is not a medical device and must not
> be used to make diagnostic or treatment decisions about real patients.

---

## 1. Decision supported

| Item | Answer |
|---|---|
| **Primary decision** | The **learner** decides, for one skin lesion: *"Is this lesion suspicious enough to refer or biopsy, or can it be reassured or monitored?"* They make this decision by applying the ABCD rule, then check their reasoning against the system's explained output. |
| **Secondary decision (curriculum)** | The **clinical educator** decides how much weight to put on teaching ABCD. They also decide which failure modes to warn students about, such as featureless melanomas and seborrheic keratoses that look like melanoma. The evidence comes from how well ABCD actually discriminates on real labelled data. |
| **Decision maker** | Medical students (primary users) and the dermatology or clinical-skills educator (curriculum owner). |
| **Cadence** | Per case, during self-study or a teaching session (tens of cases per student per session). Curriculum decisions happen once per course revision. |
| **What changes because of this work** | 1. Students get immediate, *decomposed* feedback (A, B, C, D sub-scores → weighted total → threshold → verdict), and the verdict is grounded in confirmed histopathology. 2. Educators get quantified evidence of where ABCD works and where it fails. |

## 2. Use scenario

**Narrative.** A student opens the web site and picks a case. The case comes from a
curated teaching set, or the student uploads a lesion image for self-study. Before the
answer is revealed, the student scores the lesion: A (asymmetry), B (border), C (colour),
D (differential structures or diameter). The site then shows its own analysis:

1. **Segmentation overlay.** The lesion outline the measurements were taken from.
2. **Per-criterion evidence.** For example, the asymmetry axes drawn on the image, the
   border segments flagged as abrupt, colour clusters with a swatch per colour counted,
   and the diameter in mm.
3. **The rule's arithmetic.** Stolz Total Dermoscopy Score
   `TDS = 1.3·A + 0.1·B + 0.5·C + 0.5·D`, with thresholds <4.75 benign, 4.75–5.45
   suspicious, >5.45 highly suspicious for melanoma. For clinical/TBP images, the
   clinical ABCD(E) equivalent is shown.
4. **A data-driven check of the rule.** A calibrated probability from an interpretable
   model trained on the *same* ABCD features. It is shown with per-feature contributions,
   so the student sees where the empirical model agrees or disagrees with the fixed weights.
5. **Similar confirmed cases.** The k nearest labelled lesions in ABCD-feature space,
   with their histopathology diagnoses ("cases that score like this one turned out to
   be…").
6. **Honest uncertainty and limits.** The confidence level, and a warning when the case
   falls in a region where ABCD is known to fail. One example is a nodular or amelanotic
   melanoma with a low TDS.
7. **Ground truth reveal** (curated cases only): the biopsy diagnosis, plus a comparison
   of the student's sub-scores with the system's and with expert annotations where they
   exist (PH2, derm7pt).

**What is known at decision time.** These are the *only* inputs the features may use,
because they are what a clinician has at the bedside:
- One lesion image: a dermoscopic image, or a clinical or 3D-TBP crop.
- The physical scale, when the source provides it (TBP crops are a fixed 15×15 mm;
  dermoscopy needs a scale or a known field of view to measure diameter).
- Patient age, sex, and anatomic site, which are visible or askable at the visit.
- **Not known:** the histopathology result, the lesion's later evolution (the "E" in
  ABCDE, unless sequential images exist), other lesions' biopsy results, or any field
  derived from the diagnosis (for example, ISIC `iddx_*` and `mel_thick_mm`). These
  are label leakage.

**Action following the output.** No clinical action. The output drives a *learning*
action: the student compares their scores, reads the explanation, and moves to the next
case. In a teaching session, the educator uses the aggregate confusion results to lead
a discussion.

**Which parts are data mining** (detail in §3). Extracting A, B, C, D from the image;
estimating malignancy probability from ABCD; checking whether the fixed rule's weights
and thresholds hold; retrieving similar cases; and comparing against an unconstrained
image model as an upper benchmark.

## 3. Task decomposition

| # | Sub-problem | Data-mining task (book p. 19) | Candidate target / output |
|---|---|---|---|
| T1 | Turn an image into ABCD measurements (segmentation → asymmetry index, border irregularity or abrupt-cutoff segments, number of colours, structures, diameter) | **Data reduction** (image → small interpretable feature vector). It also uses **classification** for structure detection where labels exist (derm7pt, PH2). | Continuous and ordinal A, B, C, D. Validated against expert annotations in PH2 (asymmetry, colours, network/dots/streaks) and derm7pt (7-point criteria). |
| T2 | **Test and confirm the ABCD rule as published.** Do the fixed weights and thresholds discriminate malignant from benign? | **Classification**, scored by a fixed expert model (no fitting) | Binary malignant vs benign (histopathology-confirmed where available). Output: sensitivity and specificity at 4.75 and 5.45, ROC/AUC. |
| T3 | Learn the *empirical* ABCD model. Which criteria actually carry the signal, and are Stolz's weights about right? | **Class-probability estimation** with an interpretable model (logistic regression, GAM, or a shallow tree) on the ABCD features only | P(malignant). Fitted coefficients compared with 1.3 / 0.1 / 0.5 / 0.5. |
| T4 | Quantify what ABCD misses (the cost of interpretability) | **Class-probability estimation** with an unconstrained model (gradient boosting on all metadata, or a CNN on pixels) as an upper benchmark only | P(malignant). The gap to T2/T3 sets the teaching message ("ABCD captures X% of the attainable discrimination"). |
| T5 | Show "cases like this one" with confirmed outcomes | **Similarity matching** (kNN in standardised ABCD space) | Ranked list of labelled neighbours and their diagnoses |
| T6 | Characterise where ABCD fails, by subtype, site, age, and image type | **Profiling** / error analysis (and **clustering** of false negatives) | Error profiles, e.g. "low-TDS melanomas are mostly nodular or on the head and neck" |

The tasks combine as follows. T1 feeds T2, T3, T5, and T6. T2 and T3 produce the
explained verdict. T4 is the benchmark only and is **never** shown to students as the
explanation. T5 and T6 supply the case-based and failure-mode explanations.

## 4. Expected-value sketch and success criteria

### 4a. Outcome values (screening framing; rough, stated assumptions)
This tool is educational, but the rule it teaches is used for triage, so errors are
framed in clinical terms:

| Outcome | Clinical meaning | Relative cost (assumed) |
|---|---|---|
| TP | Melanoma or other skin cancer referred | Benefit. Early excision of thin melanoma has a 5-year survival above 95%. |
| FN | Malignancy reassured | **Very high.** Delayed diagnosis. Assumed ≈ 50–100× the cost of an FP. |
| FP | Benign lesion biopsied | Low to moderate. A biopsy is roughly US$150–400, plus a scar and anxiety. |
| TN | Benign lesion reassured | Benefit of avoiding an unnecessary procedure |

**Implication.** The operating point favours **high sensitivity**. Results are reported at
fixed high-sensitivity points (80%, 90%, 95% TPR). We also report the number needed to
excise (NNE) at those points, as a clinically intuitive false-alarm measure. This matches
the ISIC 2024 metric (partial AUC above 80% TPR).

### 4b. Quantitative success criteria (to be tested in Stage 5 on a patient-grouped holdout)
| ID | Criterion | Target |
|---|---|---|
| S1 | **Rule confirmation (T2).** Sensitivity and specificity of TDS at the published thresholds, with 95% bootstrap CIs, on dermoscopic data | Reported against published figures (literature values to be collected in Stage 2). "Confirmed" if the CI overlaps the published range; otherwise the deviation is documented and explained. This is a scientific finding either way, not pass/fail. |
| S2 | Interpretable ABCD model (T3) beats the trivial baseline and the fixed rule | ROC-AUC > fixed-rule AUC, and both significantly above 0.5. pAUC above 80% TPR reported. |
| S3 | Interpretability cost (T4 − T3) | Gap in AUC and pAUC reported with CIs. No target. It is the teaching message. |
| S4 | Feature extraction validity (T1) | Agreement with expert annotations: Cohen's κ ≥ 0.4 (moderate) for asymmetry and colour count on PH2. Correlation ≥ 0.5 with the SLICE-3D TBP precomputed border and colour indices. |
| S5 | Calibration of displayed probabilities | Expected calibration error ≤ 0.05 on the holdout (after recalibration for base rate) |
| S6 | Latency of the web demo | ≤ 5 s from case selection or upload to full explanation, on a laptop CPU |

### 4c. Qualitative success criteria
- **Comprehensibility (must-have).** Every verdict on the site can be traced to visible
  sub-scores, weights, the threshold, and the evidence overlays. No black-box score is
  ever shown as "the diagnosis".
- **Pedagogical validity.** A clinical educator reviews 20 sample explanations and judges
  them clinically correct and useful. Optional pilot: a pre/post quiz with a small group
  of students (needs a separate ethics/IRB decision, recorded as an open checkpoint).
- **Catastrophic-error tolerance.** Zero tolerance for (a) presenting the tool as
  diagnostic, (b) leaking label-derived fields into features, and (c) silently
  reassuring a case in a known ABCD blind spot. The site must show the blind-spot warning.
- **Data licence compliance.** ISIC data is CC-BY-NC (to be verified per dataset in
  Stage 2). Attribution must be shown, the site must be non-commercial, and no raw
  licensed data may be committed to the repo.

### 4d. Stop criteria
- If T1's extracted features cannot reach S4 on any dataset, fall back to expert or
  precomputed features (PH2 annotations, SLICE-3D TBP features) and teach from those.
  This is loop 3 → 2.
- If neither T2 nor T3 beats chance (S2 fails), stop and return to Stage 1. That result
  would itself be a teaching finding, but the site's framing would have to change.

## 5. Stakeholders and sign-off

| Role | Who | Signs off on |
|---|---|---|
| Project owner | User (dantongyu) | Formulation (end of Stage 1), go/no-go at Stage 5 |
| Clinical content reviewer | **TBD:** a dermatologist or dermatology educator | Correctness of ABCD scoring and explanations, blind-spot warnings (Stage 5) |
| Curriculum owner | **TBD:** a course director for the target students | Fit for teaching use, any student pilot |
| Ethics / IRB | **TBD:** only if a student-learning study is run | Study protocol |
| Deployment partner | User, or a web developer (static or small server-side app) | Hosting, data-licence compliance, disclaimer placement (Stage 6) |

## 6. Assumptions and open questions for Stage 2
1. **Which "ABCD"?** There are two distinct rules:
   - **Dermoscopic ABCD (Stolz TDS).** Asymmetry, Border, Colour, Differential structures.
     It needs dermoscopic images. Candidate data: PH2 (200 images, expert-annotated
     asymmetry, colours, and structures), derm7pt (about 1,000 cases, 7-point criteria),
     HAM10000 / ISIC 2019–2020 (large, labels only).
   - **Clinical ABCDE.** Asymmetry, Border, Colour, Diameter > 6 mm, Evolving. It fits
     **ISIC 2024 SLICE-3D**, which is 3D-TBP crops, *not* dermoscopy. SLICE-3D ships
     precomputed TBP Lesion-Visualizer features that map to ABCD (`tbp_lv_symm_2axis`,
     `tbp_lv_norm_border`, `tbp_lv_norm_color`, `clin_size_long_diam_mm`, …).

   **Proposal:** cover both. One module is "clinical ABCD(E) on SLICE-3D" (large, real
   prevalence, metric precomputed). The other is "dermoscopic ABCD/TDS on PH2 + derm7pt"
   (expert ground truth for each criterion). *Needs user confirmation.*
2. SLICE-3D: confirm the label definition (`target`, histopathology confirmation rate),
   the extreme imbalance (on the order of 0.1% malignant), the patient-ID field for
   grouped splits, the licence, and that the hidden-test labels are unavailable. That
   last point means our holdout must be carved from train by patient.
3. Whether a physical scale exists in each dermoscopic dataset (this decides whether D =
   Diameter can be computed).
4. Published sensitivity and specificity for the Stolz TDS, for comparison in S1.
5. Whether a clinical reviewer and real students are available (this affects the Stage 5
   sign-off and the pilot).
6. Hosting target: a local demo only, or a public non-commercial site (which affects
   licence and attribution handling).

## 7. Iteration log
| Version | Date | Change | Why |
|---|---|---|---|
| 1.0 | 2026-10-01 | Initial formulation | Project start |
| 1.1 | 2026-10-01 | User confirmed: scope = both modules (clinical ABCD(E) on SLICE-3D + dermoscopic TDS on PH2/derm7pt); success criteria S1–S6 + qualitative accepted as written; hosting = local demo | Stage 1 human checkpoint |
| 1.2 | 2026-10-01 | Primary target = melanoma vs benign (any-malignancy secondary); clinical module uses precomputed TBP features; add patient-relative ("ugly duckling") features; S4 scope depends on PH2 access; S1 benchmark = Harrington 2017 pooled sens 73–93% | Stage 2 findings (02 §5): 236/393 SLICE-3D malignancies are BCC/SCC; crops ~131 px |

## Exit-gate self-check
- [x] Every sub-problem maps to a named data-mining task (§3, T1–T6)
- [x] Use scenario states what is available at decision time (§2, incl. explicit leakage exclusions)
- [x] Success criteria are measurable and tied to the goal (§4b S1–S6, §4c)
- [x] Sign-off owners identified (§5). Two are role-identified but still need a named
      person (clinical reviewer, curriculum owner). That is required before Stage 5, not Stage 2.
