# CRISP-DM State — skin-cancer

**Project:** Test and confirm the dermoscopic ABCD rule for melanoma/skin-cancer triage on ISIC data (ISIC 2024 / SLICE-3D and others), producing explainable, case-level output for medical students, demonstrated via a web site.
**Iteration:** 2 (iteration 1 complete 2026-10-01)
**Current stage:** 1 — awaiting user (iteration 2 reformulation; see 01 'Lessons from iteration 1')

## Stage table
| Stage | Report | Last updated | Gate |
|---|---|---|---|
| 1 Business Understanding | crisp/01_business_understanding.md | 2026-10-01 | gate-passed (user confirmed; v1.2 refined from Stage 2) |
| 2 Data Understanding | crisp/02_data_understanding.md | 2026-10-01 | gate-passed |
| 3 Data Preparation | crisp/03_data_preparation.md | 2026-10-01 | gate-passed |
| 4 Modeling | crisp/04_modeling.md | 2026-10-01 | gate-passed |
| 5 Evaluation | crisp/05_evaluation.md | 2026-10-01 | gate-passed (owner sign-off; clinical review outstanding) |
| 6 Deployment | crisp/06_deployment.md | 2026-10-01 | gate-passed (local demo; case logging deferred) |

## Transition log
| Date | Iter | From → To | Reason |
|---|---|---|---|
| 2026-10-01 | 1 | start → 1 | New project created |
| 2026-10-01 | 1 | 1 → 2 | Gate passed; user confirmed both modules, criteria as written, local demo |
| 2026-10-01 | 1 | 2 → 3 | Gate passed; target refined to melanoma (01 v1.2); PH2 obtained, derm7pt dropped |
| 2026-10-01 | 1 | 3 → 4 | Gate passed; learned D detectors (user choice); demo scope n=1,981 HAM lesions (user) |
| 2026-10-01 | 1 | 4 → 5 | Gate passed; selected D3c (derm, CV AUC 0.827) and C2 (clinical, 0.833) |
| 2026-10-01 | 1 | 5 → 6 | Holdout: derm D3c AUC 0.847, PH2 0.830; S1 not confirmed for automated TDS; clinical → fixed count rule; owner approved local demo |
| 2026-10-01 | 1 | 6 deployed | Local Streamlit demo (app/demo.py); parity verified; 0 app exceptions |
| 2026-10-01 | 1→2 | 6 → 1 | Loop closed: lessons in 01 v1.3 (measurement bottleneck, screening false alarms, blind spots) |

## Open human checkpoints
- Clinical content review by a dermatologist (required before use with real students).
- Iteration 2: user to confirm reformulation (e.g. student-entered ABCD scores) before Stage 1 re-runs.
- (resolved) Stage 2 data access; Stage 5 owner sign-off 2026-10-01.
