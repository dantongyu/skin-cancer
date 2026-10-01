"""Stage 5: score the Stage-4 holdouts ONCE, with bootstrap CIs, plus PH2 external.

Writes reports/stage5/evaluation.json. Usage: uv run python -m skin_cancer.model.evaluate
"""
import io
import json
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from skin_cancer.config import DATA_DIR, REPORTS
from skin_cancer.model.train import pauc80

S4 = REPORTS / "stage4"
OUT = REPORTS / "stage5"
RNG = np.random.default_rng(0)
B = 1000
HARRINGTON = {"cut": 4.75, "sens": (0.85, 0.73, 0.93), "spec": (0.72, 0.65, 0.78)}


def boot(y, s, fn, b=B):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(b):  # stratified bootstrap keeps both classes present
        idx = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        vals.append(fn(y[idx], s[idx]))
    return [round(float(fn(y, s)), 4), round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def sens_spec(y, pred):
    y, pred = np.asarray(y), np.asarray(pred).astype(bool)
    return float((pred & (y == 1)).sum() / (y == 1).sum()), float((~pred & (y == 0)).sum() / (y == 0).sum())


def ss_ci(y, pred):
    se = boot(y, pred, lambda a, b: sens_spec(a, b)[0])
    sp = boot(y, pred, lambda a, b: sens_spec(a, b)[1])
    return {"sens": se, "spec": sp}


def nne(sens, spec, prev):
    """Number needed to excise per melanoma found at a given prevalence."""
    return round(1 + (1 - spec) * (1 - prev) / max(sens * prev, 1e-12), 1)


def spec_at_sens(y, s, target=0.85):
    y, s = np.asarray(y), np.asarray(s)
    cut = np.quantile(s[y == 1], 1 - target)
    return sens_spec(y, s >= cut)[1]


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return round(float(e), 4)


def prior_correct(p, prior):
    """Undo class_weight='balanced' (effective prior 0.5) -> training prior."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + np.log(prior / (1 - prior)))))


def dermoscopic(res):
    h = pd.read_parquet(S4 / "ham10000_holdout_predictions.parquet")
    y = h.y_melanoma.astype(int).to_numpy()
    summ = json.loads((S4 / "modeling_summary.json").read_text())["dermoscopic"]
    out = {"n": len(h), "n_melanoma": int(y.sum())}
    for col, name in [("TDS", "D1 fixed Stolz TDS"), ("p_D2", "D2 LR on A,B,C,D"),
                      ("p_D3", "D3 LR continuous"), ("p_D3c", "D3c LR continuous+context (SELECTED)"),
                      ("p_D4", "D4 benchmark GBM")]:
        out[name] = {"auc": boot(y, h[col], roc_auc_score), "pauc80": boot(y, h[col], pauc80),
                     "spec_at_sens85": round(spec_at_sens(y, h[col]), 3)}
    out["baseline_auc"] = 0.5
    # S1: published cut-offs on automated TDS vs Harrington 2017 pooled estimates
    s1 = {}
    for cut in (4.75, 5.45):
        s1[f"TDS>{cut}"] = ss_ci(y, h.TDS > cut)
    cut_tr = summ["tds_recalibrated_cut_sens85_train"]
    s1[f"TDS>={cut_tr} (recalibrated on train)"] = ss_ci(y, h.TDS >= cut_tr)
    s1["harrington_2017_pooled_TDS>4.75"] = HARRINGTON
    out["S1_rule_confirmation"] = s1
    # S5 calibration of the selected model
    prior = summ["mel_train"] / summ["n_train"]
    out["S5_calibration_D3c"] = {"ece_raw_balanced": ece(y, h.p_D3c),
                                 "ece_prior_corrected": ece(y, prior_correct(h.p_D3c, prior)),
                                 "train_prior": round(prior, 3)}
    # business: NNE at referral prevalence (HAM10000 lesion-level melanoma rate 614/7470)
    prev = 614 / 7470
    ss_tds = sens_spec(y, h.TDS >= cut_tr)
    cut85 = np.quantile(h.p_D3c[y == 1], 0.15)
    ss_d3c = sens_spec(y, h.p_D3c >= cut85)
    out["business_NNE_at_prev_0.082"] = {
        "TDS_recalibrated": {"sens": round(ss_tds[0], 3), "spec": round(ss_tds[1], 3), "NNE": nne(*ss_tds, prev)},
        "D3c_at_holdout_sens85_point(descriptive)": {"sens": round(ss_d3c[0], 3), "spec": round(ss_d3c[1], 3),
                                                     "NNE": nne(*ss_d3c, prev)}}
    # T6 error profile: melanomas missed by D3c at the sens-85 point
    fn = h[(y == 1) & (h.p_D3c < cut85)]
    tp = h[(y == 1) & (h.p_D3c >= cut85)]
    out["T6_missed_melanoma_profile"] = {
        "n_missed": len(fn), "mean_ABCD_missed": fn[["A", "B", "C", "D", "TDS"]].mean().round(2).to_dict(),
        "mean_ABCD_detected": tp[["A", "B", "C", "D", "TDS"]].mean().round(2).to_dict()}
    res["dermoscopic_holdout"] = out

    # PH2 external
    p = pd.read_parquet(S4 / "ph2_external_predictions.parquet")
    yp = (p.clinical_dx == 2).astype(int).to_numpy()
    ext = {"n": len(p), "n_melanoma": int(yp.sum())}
    for col in ("TDS", "p_D2", "p_D3"):
        ext[col] = {"auc": boot(yp, p[col], roc_auc_score)}
    ext["TDS>4.75"] = ss_ci(yp, p.TDS > 4.75)
    ext[f"TDS>={cut_tr}"] = ss_ci(yp, p.TDS >= cut_tr)
    res["ph2_external"] = ext


def clinical(res):
    h = pd.read_parquet(S4 / "isic2024_holdout_predictions.parquet")
    y = h.y_melanoma.astype(int).to_numpy()
    out = {"n": len(h), "n_melanoma": int(y.sum()), "prevalence": round(float(y.mean()), 5)}
    for col, name in [("D_gt6mm", "C1 diameter>6mm"), ("count_rule", "C1b ABCD count"),
                      ("p_C2", "C2 LR ABCD core (SELECTED)"), ("p_C3c", "C3c LR +ugly+context"),
                      ("p_C4", "C4 benchmark GBM")]:
        out[name] = {"auc": boot(y, h[col].astype(float), roc_auc_score, b=300),
                     "pauc80": boot(y, h[col].astype(float), pauc80, b=300)}
    se, sp = sens_spec(y, h.D_gt6mm == 1)
    out["D>6mm_operating_point"] = {"sens": round(se, 3), "spec": round(sp, 3),
                                    "flagged_per_100k_crops": round(float((h.D_gt6mm == 1).mean() * 1e5)),
                                    "NNE_at_holdout_prevalence": nne(se, sp, y.mean())}
    cr = h.count_rule >= 2
    se2, sp2 = sens_spec(y, cr)
    out["count>=2_operating_point"] = {"sens": round(se2, 3), "spec": round(sp2, 3),
                                       "NNE_at_holdout_prevalence": nne(se2, sp2, y.mean())}
    st = h[h.strong_label == 1]
    ys = st.y_melanoma.astype(int)
    out["strong_label_subset(mel vs biopsied benign)"] = {
        "n": len(st), "n_melanoma": int(ys.sum()),
        "auc_C2": round(float(roc_auc_score(ys, st.p_C2)), 3),
        "auc_diam": round(float(roc_auc_score(ys, st.clin_size_long_diam_mm)), 3)}
    by_site = {}
    for site, g in h.groupby("attribution"):
        if g.y_melanoma.sum() >= 5:
            by_site[site[:40]] = {"n_mel": int(g.y_melanoma.sum()),
                                  "auc_C2": round(float(roc_auc_score(g.y_melanoma, g.p_C2)), 3)}
    out["by_institution(n_mel>=5)"] = by_site
    tiny = h[(y == 1) & (h.clin_size_long_diam_mm <= 2.74)]
    out["tiny_melanomas_in_holdout"] = {"n": len(tiny),
                                        "C2_rank_percentile": [round(float((h.p_C2 < v).mean()), 3) for v in tiny.p_C2]}
    res["clinical_holdout"] = out


def latency(res):
    from skin_cancer.prep.abcd_image import extract
    from skin_cancer.prep.dermoscopy import load_thresholds
    from skin_cancer.prep.structure_cnn import lesion_crop, load_trained, predict
    z = zipfile.ZipFile(DATA_DIR / "ph2" / "PH2.zip")
    rgb = np.array(Image.open(io.BytesIO(z.read("ph2_dataset/trainx/IMD403.bmp"))).convert("RGB"))
    m = np.array(Image.open(io.BytesIO(z.read("ph2_dataset/trainy/IMD403_lesion.bmp"))).convert("L")) > 127
    model, _ = load_trained()
    t = load_thresholds()
    predict(model, lesion_crop(rgb, m)[None])          # warm-up
    t0 = time.time()
    extract(rgb, m, t, keep_evidence=True)
    predict(model, lesion_crop(rgb, m)[None])
    joblib.load(DATA_DIR / "models" / "dermoscopic_models.joblib")
    res["S6_latency_seconds_one_case"] = round(time.time() - t0, 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    dermoscopic(res)
    clinical(res)
    latency(res)
    (OUT / "evaluation.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
