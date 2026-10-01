# 06 — Deployment

**Project:** skin-cancer · **Iteration:** 1 · **Version:** 1.0 · **Date:** 2026-10-01
**Approval:** owner sign-off for a **local educational demo** (05 §5). Clinical review is outstanding.

## 1. Deployment form
This deploys the **model inside a teaching process**: a local web app, `app/demo.py` (Streamlit, served on `localhost:8501` only).

Run it with:
```
uv run streamlit run app/demo.py
```

There are no external services, no real patients and no public hosting. That keeps it within the licences (CC-BY-NC, and PH2 local-only) and within the sign-off.

| Page | What the student does or sees |
|---|---|
| **Dermoscopic case walk-through** | **Choose a case:** a held-out HAM10000 case (never used in training), a PH2 case, or an uploaded image (auto-segmented). Then: (0) score A/B/C/D yourself; (1) lesion outline; (2) per-criterion evidence: asymmetry axes, abrupt border sectors, colour map with low-agreement warnings, structure Grad-CAM heat-maps and the structureless area; (3) the TDS arithmetic, with the published cut-offs *and* the finding that they don't transfer to automated scoring; (4) the interpretable model's probability and per-feature contributions; (5) five most similar confirmed training cases; (6) blind-spot warnings; (7) reveal the diagnosis and compare your scores with the system's (and the PH2 expert's) |
| **Clinical TBP panel** | A random held-out ISIC 2024 crop with its A/B/C/D measurements, flags, ABCD count and the > 6 mm rule, plus the false-alarm (NNE ≈ 376) lesson. Then reveal |
| **Project results** | Holdout AUCs with CIs, and the S1 rule-confirmation table vs Harrington 2017 |

A disclaimer ("Educational prototype — not clinically reviewed, not a medical device") is on every page.

## 2. Hand-off package
| Item | Location |
|---|---|
| Prototype | `app/demo.py` |
| Preparation pipeline | `src/skin_cancer/prep/` (`isic2024.py`, `abcd_image.py`, `dermoscopy.py`, `structure_cnn.py`) |
| Thresholds and provenance | `reports/stage3/dermoscopy_thresholds.json` |
| Models | `~/data/skin-cancer/models/`: `structure_resnet18.pt`, `structure_thresholds.json`, `dermoscopic_models.joblib`, `clinical_models.joblib` |
| Data dictionaries | `reports/stage3/*_data_dictionary.csv` |
| Evaluation | `crisp/05_evaluation.md`, `reports/stage5/evaluation.json` |
| Reproduce from scratch | README "Reproduce" section |

Owner: project owner (dantongyu). There is no separate engineering team; this is a local prototype.

## 3. Parity tests (2026-10-01)
| Test | Expected | Observed |
|---|---|---|
| Holdout case `ISIC_0031306` through the app pipeline vs the Stage 4 stored output | A, B, C, D = 0, 0, 2, 2; TDS 2.0; raw p_D3c 0.073 | **0, 0, 2, 2; TDS 2.0**; prior-corrected 0.034, which is the corrected value of raw 0.073 ✔ |
| Latency, one case (app pipeline incl. CNN, M2) | ≤ 5 s | **0.90 s** (Stage 5 measured 0.67 s for the core pipeline) ✔ |
| Headless app test (`streamlit.testing`): all 3 pages + the full 8-step analysis | No exceptions | **0 exceptions** ✔ |
| Server health (`/_stcore/health`) | ok | ok ✔ |

## 4. Monitoring, fail-safes, rollback
- **Fail-safes:**
  - Lesion masks under 200 px raise an error instead of producing a score (`extract`).
  - Uploads are labelled as auto-segmented.
  - Red and white colours carry a ⚠️ low-agreement mark.
  - The B criterion carries a "no ground truth" note.
  - The clinical module shows rules only, never probabilities.
- **Monitoring** (05 §7, to add in iteration 2): a local CSV log of cases shown and student vs system scores; alerts for segmentation failure, Lab shift > 25, and saturated CNN outputs.
- **Rollback:** the demo is stateless and local. Stop the server, or check out an earlier git commit. Models are versioned by filename in `~/data/skin-cancer/models/`.

## 5. Rollout
- **Status:** running locally for the project owner (pilot).
- **Audience (owner, 2026-10-01):** the owner and a few collaborators, as a research demo. No student use is planned, so the dermatologist review and student pilot are descoped.
- **Sharing:** demo it by screen-share, or each collaborator runs it locally with their own downloads (ISIC/HAM10000 are CC-BY-NC, so non-commercial use is fine). **Do not pass on PH2 files** (no redistribution); collaborators can obtain PH2 themselves. Exposing the app on a network or a public URL would be a new deployment step and needs explicit approval.

## 6. Lessons learned → iteration 2 (fed into 01 v1.3)
1. **Measurement, not the clinical rule, is the bottleneck.** Expert A and C discriminate (AUC 0.86 / 0.87), but automated C (κ ≤ 0.23) and B (no ground truth) don't. Published TDS cut-offs fail on automated scores (sensitivity 8%). Options for iteration 2:
   - better colour and border measurement;
   - a hybrid demo where **students enter A/B/C/D** and the system evaluates *their* TDS;
   - collect expert B labels.
2. **Continuous measurements beat Stolz points** (AUC 0.79 vs 0.70). Teaching should show both.
3. **ABCD is not a screening rule.** In total-body photography it costs about 376 excisions per melanoma. Teach it for lesions that already stand out, together with the ugly-duckling sign and evolution.
4. **Blind spots to teach:** symmetric melanomas and melanomas ≤ 3 mm.
5. **Engineering:**
   - Use the full ISIC 2024 training set.
   - Separate dots and globules (needs labels).
   - Patient-level splits for HAM10000.
   - Tune the structure CNN's learning rate (peak 1e-3 destabilised early epochs).
   - Add a pixel-CNN benchmark.
6. **Process:** the demo-scope shortcuts (subsampling, one image per lesion) were the right call for a prototype, but they must be lifted before any published claims.

## Exit gate
- [x] Parity verified (§3); owner named (§2); rollback defined and trivially testable (stop or checkout).
- [~] Monitoring: fail-safes are live, but case logging is planned for iteration 2. Acceptable for a local pilot without real users.
- [x] Lessons fed back into `01_business_understanding.md` (v1.3 iteration-log entry). STATE moves to iteration 2, Stage 1.
