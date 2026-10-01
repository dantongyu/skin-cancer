# 02 — Data Understanding

**Project:** skin-cancer · **Iteration:** 1 · **Version:** 1.1 · **Date:** 2026-10-01
**Inputs:** `crisp/01_business_understanding.md` v1.2 (both modules; local demo)

Every number below comes from a saved output in `reports/stage2/`:
- `isic2024_profile.json`
- `isic2024_abcd_proxy_auc.csv`
- `isic2024_abcd_proxy_auc_melanoma.csv`
- `isic2024_missingness.csv`
- `isic2024_image_sizes.txt`
- `isic2024_melanoma_patients.txt`
- `dermoscopy_profile.json`

These are produced by `src/skin_cancer/data/profile_isic2024.py` and
`src/skin_cancer/data/profile_dermoscopy.py`. Literature and licence facts come from a
web review dated 2026-10-01; the sources are cited inline. Raw data lives in
`~/data/skin-cancer/` (`SKIN_DATA_DIR`), **outside the repo**.

---

## 1. Source inventory

| Source | Contents | Origin / purpose | Population | Reliability | Cost to obtain | Decision |
|---|---|---|---|---|---|---|
| **ISIC 2024 SLICE-3D** (train) | 401,059 15×15 mm crops from 3D total-body photos. **55 columns**: age, sex, site, `patient_id`, 34 Canfield Lesion Visualizer (`tbp_lv_*`) features incl. ABCD proxies, diagnosis hierarchy `iddx_*` | Kaggle/ISIC 2024 challenge. Crops were auto-extracted from Vectra WB360 scans at 7 centres (Kurtansky et al., *Sci Data* 2024, doi:10.1038/s41597-024-03743-w) | 1,042 patients, median age 60 (`age_approx`); 7 institutions in the US, Europe and Australia | Malignant labels are histopathology-confirmed ("strong"). Benign labels are mostly *weak*: assumed benign clinically, no biopsy (see §2) | Free; downloaded (1.24 GB + 49 MB CSVs). **CC-BY-NC 4.0** (LICENSE.txt inside the zip); individual images carry CC-0, CC-BY or CC-BY-NC | **USE.** Clinical ABCD(E) module (T2–T6) |
| **HAM10000** | 10,015 dermoscopic images, 7,470 lesions, 7 diagnoses, age, sex, site. 10,016 lesion masks (Tschandl) | Collected in Vienna and Queensland for training and benchmarking classifiers (Tschandl 2018) | Referral or specialist population, **enriched for disease** (melanoma is 614 of 7,470 lesions) | Melanoma: 614/614 histopathology. Nevi: 1,367 histo, 3,704 follow-up, 332 consensus (lesion level) | Free (Harvard Dataverse doi:10.7910/DVN/DBW86T); metadata, masks and images (2.8 GB) downloaded. **CC-BY-NC 4.0** | **USE.** Dermoscopic module: extract ABCD from images (T1), compute TDS (T2), plus T3–T6 |
| **ISIC 2018 Task 2** attribute masks | 2,594 dermoscopic images × 5 attribute masks | Challenge task on detecting dermoscopic structures (Codella 2019) | ISIC Archive images | Expert-drawn masks | Ground truth (35 MB) downloaded, CC-BY-NC. Images: 11.2 GB, CC-0, *not yet downloaded* | **USE** to validate the detectors for the D criterion (T1). Image download is a Stage 3 decision |
| **PH2** | 200 dermoscopic images (80 common nevi, 80 atypical nevi, 40 melanoma), 768×560 at 20×. Expert **asymmetry (0/1/2), colours (6-set), network, dots/globules, streaks, regression, blue-white veil**, lesion masks | Univ. Porto / Hosp. Pedro Hispano (Mendonça 2013), for CAD research | Single centre | Label is the *clinical* diagnosis; histology exists for 41/200 images (33/40 melanomas, 8/80 atypical nevi). **No border (0–8) score**, so a full ground-truth TDS is impossible | Official Dropbox link is dead. **User approved an unofficial mirror on 2026-10-01**: images + masks from Zenodo record 17498821 (200 + 200, IDs all match), annotations from GitHub `vikaschouhan/PH2-dataset/PH2_dataset.txt`. Terms are "research and educational purposes… redistribution not allowed", so the data stays local and is never committed or redistributed | **USE** (local only). Validates our asymmetry and colour measurements (S4) |
| **derm7pt** (SFU) | 1,011 cases, clinical and dermoscopic images, 7-point checklist criteria, diagnosis, management, official train (413) / valid (203) / test (395) split | Kawahara 2019, for multi-task CAD | Atlas cases (Argenziano), specialist | Expert criteria labels. No diameter, no asymmetry, no colour count | Free, but **needs registration via a Google form** (https://forms.gle/iPnCEVkUKYwWYALh9). **CC BY-NC-ND 4.0** (no derivatives: fine for a local demo; overlays must not be redistributed) | **DROP** (user decision 2026-10-01). Not needed for this iteration |
| ISIC 2018 Task 1 | 2,594 lesion masks | Segmentation challenge | ISIC Archive | Expert masks | 11.2 GB images | **DROP.** The HAM10000 masks are enough for segmentation |
| ISIC 2024 hidden test | 500k+ crops | Kaggle leaderboard | n/a | n/a | **Not released** (ground-truth CSV returns 404) | **DROP.** Our holdout must come from train, grouped by patient |

## 2. Target-label assessment

### ISIC 2024 SLICE-3D (clinical module)
- `malignant` (= Kaggle `target`): **393 / 401,059 = 0.098 %**, from 259 of 1,042 patients.
  The malignant set is **163 BCC, 73 SCC, 157 melanoma** (80 in situ, 63 invasive,
  13 NOS, 1 metastasis).
- **ABCD is a melanoma rule.** BCC and SCC have different clinical signs, so the
  primary target is changed to **melanoma vs benign**: 157 melanomas in 133 patients,
  base rate 0.039 %. *Any malignancy* is kept as a secondary target. This refines
  Stage 1 (logged in 01 v1.2).
- **Label reliability.**
  - Every malignant crop has a histopathology report within 3 months of capture.
  - Of 400,666 benign crops, **399,991 are plain "Benign"** with no further diagnosis.
    These are *weak* labels: judged benign clinically and never biopsied. Only 675
    benign crops have a histopathology subtype.
  - Weak benign labels carry a small risk of unbiopsied melanomas, and we judge this
    acceptable. It also means "benign" here means "not suspicious enough to biopsy",
    so the benign class is easy and AUC will look optimistic.
  - Evaluation should also report **melanoma vs strongly-labelled benign**: 675
    crops have a subtype, minus 114 indeterminate = **561**, mostly atypical and
    other nevi. That is the clinically hard comparison.
- **Indeterminate.** 114 crops (64 atypical melanocytic neoplasm, 11 AIMP, 39 actinic
  keratosis) have `malignant = 0`. They are **excluded from the primary analysis**
  because they are ambiguous.
- **Positives are tiny in absolute terms.** 157 melanomas in 133 patients. A 20 %
  patient-grouped holdout holds only about 30 melanomas, so sensitivity CIs will be
  roughly ±15 pp. We plan grouped cross-validation plus a holdout, and report CIs
  everywhere.

### HAM10000 (dermoscopic module)
- `dx`: melanoma 614 lesions, all histopathology. The TDS was designed for
  **melanocytic** lesions, so the primary comparison is **melanoma vs nevus** (614 vs
  5,403 lesions). The secondary comparison is melanoma vs all benign (adds bkl, df, vasc).
- **Label reliability.** Nevus labels: 1,367 histo, 3,704 follow-up (benign by
  stability), 332 consensus. Reliable, but the follow-up nevi are "easy". As with
  ISIC 2024, we will add a **histo-only** sensitivity analysis.
- **No patient ID**, only `lesion_id` (up to 6 images per lesion). Splits must group by
  `lesion_id`. Two lesions from the same patient can still land on both sides of a
  split, which is a residual leak risk.
- **Prevalence is enriched** (about 10 % melanoma vs 0.04 % in TBP screening). PPV and
  NNE must be re-weighted to a screening base rate before being shown to students.

### Labels for each ABCD criterion (needed for T1 and S4)
| Criterion | Ground truth available |
|---|---|
| A asymmetry 0/1/2 | **PH2 only** (pending). SLICE-3D has an AI-estimated `tbp_lv_symm_2axis`, but those are TBP crops, not dermoscopy |
| B border 0–8 | **None** in any public dataset. SLICE-3D only has the AI proxies `norm_border` and `area_perim_ratio`. An automated B score can be checked only indirectly (contribution to discrimination, visual review by the clinician) |
| C colours | **PH2 only** (pending) |
| D structures | ISIC 2018 Task 2: pigment network in 1,523 images, globules 603, negative network 190, streaks **100** (rare), milia-like cysts 682. Also PH2 and derm7pt (pending). Stolz's "structureless areas" and "dots" have no direct labels |
| D diameter (clinical ABCDE) | SLICE-3D `clin_size_long_diam_mm` (calibrated mm) |

## 3. Profiling summary

### SLICE-3D
- Images: **median 131 px** across a 15 mm crop, about 0.115 mm/px (500-image sample).
  That is too coarse to detect dermoscopic structures. The clinical module will
  therefore use the **precomputed Lesion Visualizer features** rather than measuring
  from pixels. Pixels are used only for display and the T4 benchmark.
- **Missingness** is low among usable features: sex 2.9 %, site 1.4 %, age 0.7 %. The
  `tbp_lv_*` features are complete. `lesion_id` is 94.5 % missing, `iddx_2–5` over
  99 %, `mel_*` over 99.9 %.
- **Signal of each ABCD proxy on its own, melanoma vs benign** (AUC in the best
  direction; `isic2024_abcd_proxy_auc_melanoma.csv`):

| Criterion | Best proxy | AUC | Comment |
|---|---|---|---|
| A | `tbp_lv_symm_2axis` | 0.62, **but reversed**: melanomas score *more* symmetric (raw AUC 0.376) | Counter to the textbook. Possibly the index's definition, or low resolution. **Investigate in Stage 3. It is also a teaching point** |
| B | `tbp_lv_norm_border` | 0.53 | About chance. `perimeterMM` reaches 0.80 but mostly encodes size |
| C | `tbp_lv_norm_color` | 0.80 | Strong |
| C | `tbp_lv_deltaLBnorm` (contrast) | 0.77 raw → **0.80 patient-relative** | The "ugly duckling" normalisation helps |
| D | `tbp_lv_areaMM2` / long diameter | 0.82 / 0.80 | Strong |

- **The clinical "D > 6 mm" rule for melanoma:** sensitivity 0.61, specificity 0.90.
  Melanoma long diameter has a median of 6.98 mm, IQR 4.93–10.35, minimum 1.07 mm. So
  **39 % of melanomas are ≤ 6 mm**: a concrete limitation worth teaching.
- **Prevalence by subgroup:** head/neck 0.65 % vs 0.06–0.09 % elsewhere; male 0.103 %
  vs female 0.088 %; by institution, from 0.020 % (Basel) to 0.156 % (Queensland).
  The institution is a likely **shortcut** and is not known at the bedside, so it is
  excluded as a feature and used for stratified or leave-one-site-out checks instead.
- **Leakage found:**
  - `lesion_id` is present on **all 393** malignant crops but only **21,665 / 400,666**
    benign ones. That is near-perfect leakage, so it is excluded.
  - `iddx_*`, `mel_thick_mm` and `mel_mitotic_index` come after the diagnosis and are
    excluded.
  - `tbp_lv_nevi_confidence` and `tbp_lv_dnn_lesion_confidence` are black-box DNN
    outputs (AUC 0.65 / 0.69 for malignancy when flipped). They are allowed **only in
    the T4 benchmark**, never in the explainable ABCD model.

### PH2 (expert annotations; `dermoscopy_profile.json` → `ph2`)
- **Asymmetry 2 (fully asymmetric):** 33/40 melanomas, 18/80 atypical nevi, 1/80
  common nevi.
- **Colour count:** mean 3.15 in melanoma vs 1.86 in atypical and 1.61 in common nevi.
- **Expert criterion alone, melanoma vs all nevi:** AUC is **0.859 for asymmetry** and
  **0.870 for colour count**. On dermoscopy with expert scoring, A and C clearly
  discriminate. That contrasts with the reversed TBP asymmetry proxy (§3, SLICE-3D)
  and is a key teaching contrast between modality and scorer.
- **Structures:** atypical network 39/40 melanomas vs 77/80 atypical nevi, so it
  separates nevus types more than melanoma from atypical nevus. Blue-white veil
  30/40 vs 6/80. Regression 21/40 vs 4/80. Streaks 13/40 vs 16/80.

### HAM10000
- 10,015 images, 7,470 lesions, 4 source collections. Age is missing for 57 images
  (median 50). There is a mask for every image, which supports pixel-level ABCD
  extraction (T1).

## 4. Linkage and collation plan
- **SLICE-3D:** join `metadata.csv` (inside the zip), the GroundTruth and the
  Supplement on `isic_id`. This has already been done (one-to-one, validated). Group
  by `patient_id` for splits and for patient-relative ("ugly duckling") features.
- **HAM10000:** join the images, masks and metadata on `image_id`. Group by `lesion_id`.
  One image per lesion for evaluation (the first), or aggregate per lesion.
- **Overlap risk across datasets:** HAM10000 images are part of the ISIC Archive and may
  overlap with ISIC 2018 Task 2 images. Before using Task 2 for detector validation,
  de-duplicate by `isic_id` so a detector is never validated on images it was tuned on.
- **The two modules are never pooled.** They use different modalities and different
  targets, and are evaluated separately.

## 5. Changes to the Stage 1 formulation (logged in 01 v1.2)
1. **Primary target is melanoma vs benign**; any-malignancy (incl. BCC and SCC) is
   secondary. Reason: ABCD and TDS are melanoma rules. 236 of the 393 SLICE-3D
   malignancies are BCC or SCC.
2. **Clinical module uses the precomputed TBP features**, not pixel measurement.
   Reason: crops are about 131 px across 15 mm.
3. **New candidate feature family: patient-relative ("ugly duckling") features.**
   They are known at decision time in TBP and are a recognised clinical sign.
4. **S4 (feature-extraction validity) uses PH2** (obtained 2026-10-01 from a mirror,
   user-approved) for asymmetry and colour, and ISIC 2018 Task 2 for structures.
   B has no ground truth anywhere.
5. **Published TDS benchmarks for S1:**
   - Nachbar 1994, n=172: sensitivity 92.8 / specificity 90.3 at >5.45.
   - Harrington 2017 meta-analysis, 8 studies: pooled 85 (73–93) / 72 (65–78) at >4.75.
   - Argenziano 1998, n=342: 85 / 66.
   - Ahnlide 2016, bedside, n=309: 83 / 45 at 4.75 and 74 / 67 at 5.45.

   Inter-observer κ for asymmetry is 0.35–0.49 (Rodríguez-Lomba 2022, PMC9133296).
   "Confirmed" in S1 means our CI overlaps the 73–93 % pooled sensitivity range.

## 6. Gaps and decisions carried forward
| Gap | Impact | Revisit at |
|---|---|---|
| No public border (0–8) ground truth | The B score can't be validated directly | Stages 3, 5 (clinician review) |
| PH2 obtained from an unofficial mirror (user-approved) | Licence: local use only, never redistribute; small (n=200, 40 melanoma) | Stage 6: the demo must not ship PH2 images publicly |
| derm7pt dropped | No 7-point checklist comparison this iteration | Next iteration |
| Weak benign labels (SLICE-3D) and follow-up nevi (HAM10000) | Optimistic AUC | Stage 5: strong-label-only analysis |
| Enriched prevalence (HAM10000) vs screening (SLICE-3D) | PPV and NNE not transferable | Stage 5: re-weight to base rate |
| No patient ID in HAM10000 | Residual leakage across lesions | Stage 4 (note in limitations) |
| Inverted asymmetry proxy in SLICE-3D | Rule "A" may not be confirmed on TBP | Stages 3–5 |
| About 30 melanomas per SLICE-3D holdout | Wide CIs | Stages 4, 5 (grouped CV + bootstrap) |
| Institution-level prevalence differences | Shortcut risk | Stage 5: leave-one-site-out |

## 7. Iteration log
| Version | Date | Change | Why |
|---|---|---|---|
| 1.0 | 2026-10-01 | Initial data understanding | First pass |
| 1.1 | 2026-10-01 | PH2 → USE (mirror, local only) + PH2 profile; derm7pt → DROP | User decisions at Stage 2 data-access checkpoint |

## Exit-gate self-check
- [x] **Each task has a data source and a credible target.**
  - T1: HAM10000 masks + Task 2 structure masks + PH2 expert asymmetry and colours.
  - T2 and T3: SLICE-3D (histo melanoma) and HAM10000 (histo melanoma).
  - T4: same data, pixels and all metadata.
  - T5 and T6: same labelled sets.
- [x] **Cost/benefit decision recorded for each source** (§1: USE ×4, DROP ×3).
- [x] **Coverage and reliability gaps written down for Evaluation** (§6).
