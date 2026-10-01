# Explainable ABCD Skin-Lesion Teaching Project

> **Educational use only.** This project is not a medical device and must not be
> used to diagnose or manage real patients.

The project tests and confirms the **ABCD rule** for melanoma detection. It covers two
versions of the rule:
- the dermoscopic **Stolz Total Dermoscopy Score**, `TDS = 1.3·A + 0.1·B + 0.5·C + 0.5·D`;
- the clinical **ABCD(E)** rule.

It runs on public, histopathology-labelled datasets. The end product is a **local web
demo** that walks medical students through how an ABCD-based assessment of a lesion is
built, step by step:

segmentation → per-criterion evidence → weighted score → threshold → calibrated
probability → similar confirmed cases → known blind spots.

The work follows **CRISP-DM** (Provost & Fawcett, *Data Science for Business*). Each
stage has a report under [`crisp/`](crisp/), and progress is tracked in
[`crisp/STATE.md`](crisp/STATE.md).

## Status (2026-10-01)

| Stage | Report | Status |
|---|---|---|
| 1 Business Understanding | [`crisp/01_business_understanding.md`](crisp/01_business_understanding.md) | ✅ gate passed, user-confirmed |
| 2 Data Understanding | [`crisp/02_data_understanding.md`](crisp/02_data_understanding.md) | ✅ gate passed |
| 3 Data Preparation | — | next |
| 4 Modeling | — | |
| 5 Evaluation | — | |
| 6 Deployment (web demo) | — | |

A narrative summary of findings so far is in [`docs/REPORT.md`](docs/REPORT.md).

## Datasets

| Dataset | Role | Licence |
|---|---|---|
| ISIC 2024 SLICE-3D (401k TBP crops, 157 melanomas) | Clinical ABCD(E) module | CC-BY-NC 4.0 |
| HAM10000 (10k dermoscopic images + masks) | Dermoscopic TDS module | CC-BY-NC 4.0 |
| ISIC 2018 Task 2 (attribute masks) | Validate detectors for dermoscopic structures | CC-BY-NC |
| PH2 (200 images, expert A/C/structure scores) | Validate asymmetry and colour measurement | Research/education only, **no redistribution** |

**No raw data is stored in this repository.** The data is downloaded to
`~/data/skin-cancer/`, or to a directory you set with `SKIN_DATA_DIR`.

## Reproduce

```bash
uv sync                                                   # Python 3.12 env
uv run python -m skin_cancer.data.download                # ~4.5 GB into ~/data/skin-cancer
uv run python -m skin_cancer.data.profile_isic2024        # -> reports/stage2/isic2024_*
uv run python -m skin_cancer.data.profile_dermoscopy      # -> reports/stage2/dermoscopy_profile.json
```

## Layout

```
crisp/              CRISP-DM stage reports + STATE.md (iteration log)
docs/REPORT.md      Narrative project report
reports/stageN/     Machine-generated outputs; every number in the reports comes from here
src/skin_cancer/    Code (config, data download + profiling; later features, models, web app)
```

## Data attribution

- **ISIC 2024:** International Skin Imaging Collaboration. *SLICE-3D 2024 Challenge
  Dataset*. doi:10.34970/2024-slice-3d. Kurtansky et al., *Sci Data* 2024.
- **HAM10000:** Tschandl, Rosendahl & Kittler, *Sci Data* 2018. doi:10.7910/DVN/DBW86T.
- **ISIC 2018:** Codella et al., arXiv:1902.03368. Tschandl et al., 2018.
- **PH2:** Mendonça et al., *EMBC* 2013.
