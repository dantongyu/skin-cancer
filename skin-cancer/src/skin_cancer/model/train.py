"""Stage 4 modeling for both modules.

1. Fix the holdout FIRST (saved to reports/stage4/*_split.csv); never used here
   for fitting, tuning, or reporting. Stage 5 scores it.
2. Baselines -> fixed published rules -> interpretable models -> black-box benchmark.
3. Model selection only by 5-fold (grouped) CV on the training part.
4. Every run is appended to reports/stage4/run_log.csv.
Fitted models + holdout predictions are saved for Stage 5 / the demo.

Usage: uv run python -m skin_cancer.model.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skin_cancer.config import DATA_DIR, REPORTS

OUT = REPORTS / "stage4"
PREP = DATA_DIR / "prepared"
MODELS = DATA_DIR / "models"
SEED = 0
RUNS: list[dict] = []


def pauc80(y, s) -> float:
    """ISIC 2024 metric: partial AUC above 80% TPR (range 0-0.2)."""
    max_fpr = 0.2
    v = roc_auc_score(1 - np.asarray(y), -np.asarray(s), max_fpr=max_fpr)
    return 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / 0.5 * (v - 0.5)


def make_pipe(num, cat, model):
    tf = [("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                            ("sc", StandardScaler())]), num)]
    if cat:
        tf.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return Pipeline([("pre", ColumnTransformer(tf)), ("clf", model)])


def cv_score(df, y, groups, num, cat, model, k=5):
    """Grouped stratified k-fold on TRAIN only; returns fold AUCs, pAUCs and OOF scores."""
    skf = StratifiedGroupKFold(k, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df))
    aucs, paucs = [], []
    for tr, va in skf.split(df, y, groups):
        p = make_pipe(num, cat, model).fit(df.iloc[tr], y[tr])
        s = p.predict_proba(df.iloc[va])[:, 1]
        oof[va] = s
        aucs.append(roc_auc_score(y[va], s))
        paucs.append(pauc80(y[va], s))
    return np.array(aucs), np.array(paucs), oof


def log_run(run_id, module, model, features, hyper, aucs, paucs, note=""):
    RUNS.append({"run_id": run_id, "module": module, "model": model,
                 "n_features": len(features), "features": ";".join(features),
                 "hyperparameters": json.dumps(hyper), "seed": SEED,
                 "cv_auc_mean": round(float(np.mean(aucs)), 4), "cv_auc_sd": round(float(np.std(aucs)), 4),
                 "cv_pauc80_mean": round(float(np.mean(paucs)), 4), "note": note})
    print(RUNS[-1]["run_id"], RUNS[-1]["model"], RUNS[-1]["cv_auc_mean"], RUNS[-1]["cv_pauc80_mean"], note)


def fixed_score_cv(df, y, groups, score, k=5):
    """CV-style fold scores for a fixed (unfitted) score, for comparable reporting."""
    skf = StratifiedGroupKFold(k, shuffle=True, random_state=SEED)
    a, p = [], []
    for _, va in skf.split(df, y, groups):
        a.append(roc_auc_score(y[va], score[va]))
        p.append(pauc80(y[va], score[va]))
    return np.array(a), np.array(p)


LR = dict(C=1.0, max_iter=2000, class_weight="balanced")
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
           class_weight="balanced", random_state=SEED)


# ------------------------------------------------------------- dermoscopic

DERM_ABCD = ["A", "B", "C", "D"]
DERM_CONT = ["a_shape_max", "a_color_max", "b_ramp_mean", "b_ramp_std", "b_compactness",
             "c_frac_white", "c_frac_red", "c_frac_light_brown", "c_frac_dark_brown",
             "c_frac_blue_gray", "c_frac_black", "c_L_std", "c_ab_std",
             "cnn_p_pigment_network", "cnn_p_globules", "cnn_p_streaks", "d_structureless_frac"]
DERM_BENCH = DERM_CONT + ["cnn_p_negative_network", "cnn_p_milia_like_cyst", "lesion_area_frac",
                          "a_shape_ax1", "a_shape_ax2", "a_color_ax1", "a_color_ax2", "b_ramp_min",
                          "c_L_mean", "d_net_score", "d_n_dots", "d_n_globules", "d_streak_score", "age"]
DERM_CAT = ["sex", "localization"]


def dermoscopic() -> dict:
    df = pd.read_parquet(PREP / "ham10000_model.parquet")
    df = df[df.primary_cohort == 1].reset_index(drop=True)
    y_all = df.y_melanoma.astype(int).to_numpy()
    tr_idx, ho_idx = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y_all,
                                      random_state=SEED)
    split = pd.DataFrame({"image_id": df.image_id, "lesion_id": df.lesion_id,
                          "split": np.where(np.isin(np.arange(len(df)), ho_idx), "holdout", "train")})
    split.to_csv(OUT / "ham10000_split.csv", index=False)
    tr, ho = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)
    y, g = tr.y_melanoma.astype(int).to_numpy(), tr.lesion_id.to_numpy()

    # D0 baseline: constant score
    a, p = fixed_score_cv(tr, y, g, np.zeros(len(tr)))
    log_run("D0", "dermoscopic", "baseline: constant (prevalence)", [], {}, a, p)
    # D1 fixed Stolz TDS (published weights; no fitting)
    a, p = fixed_score_cv(tr, y, g, tr.TDS.to_numpy())
    log_run("D1", "dermoscopic", "fixed rule: Stolz TDS = 1.3A+0.1B+0.5C+0.5D", DERM_ABCD,
            {"weights": [1.3, 0.1, 0.5, 0.5]}, a, p)
    # D1b recalibrated cut-off on TRAIN for sensitivity >= 0.85 (meta-analysis pooled sens)
    s = tr.TDS.to_numpy()
    cut = float(np.quantile(s[y == 1], 0.15))
    # D2 logistic regression on the 4 Stolz sub-scores (learned weights)
    a, p, _ = cv_score(tr, y, g, DERM_ABCD, [], LogisticRegression(**LR))
    log_run("D2", "dermoscopic", "logistic regression on A,B,C,D", DERM_ABCD, LR, a, p)
    # D3 logistic regression on continuous ABCD measurements
    a, p, _ = cv_score(tr, y, g, DERM_CONT, [], LogisticRegression(**LR))
    log_run("D3", "dermoscopic", "logistic regression on continuous ABCD measurements", DERM_CONT, LR, a, p)
    # D3c + context
    a, p, _ = cv_score(tr, y, g, DERM_CONT + ["age"], DERM_CAT, LogisticRegression(**LR))
    log_run("D3c", "dermoscopic", "logistic regression, continuous ABCD + age/sex/site",
            DERM_CONT + ["age"] + DERM_CAT, LR, a, p)
    # D4 black-box benchmark
    a, p, _ = cv_score(tr, y, g, DERM_BENCH, DERM_CAT, HistGradientBoostingClassifier(**HGB))
    log_run("D4", "dermoscopic", "BENCHMARK gradient boosting, all features + context",
            DERM_BENCH + DERM_CAT, HGB, a, p)

    # selection among interpretable fitted models (D2, D3, D3c): best CV AUC;
    # prefer the simpler one if within 0.02
    cand = [r for r in RUNS if r["run_id"] in ("D2", "D3", "D3c")]
    best = max(cand, key=lambda r: r["cv_auc_mean"])
    simpler = [r for r in cand if r["cv_auc_mean"] >= best["cv_auc_mean"] - 0.02]
    chosen = min(simpler, key=lambda r: r["n_features"])
    spec = {"D2": (DERM_ABCD, []), "D3": (DERM_CONT, []), "D3c": (DERM_CONT + ["age"], DERM_CAT)}

    fitted = {}
    for rid, (num, cat) in spec.items():
        fitted[rid] = make_pipe(num, cat, LogisticRegression(**LR)).fit(tr, y)
    fitted["D4"] = make_pipe(DERM_BENCH, DERM_CAT, HistGradientBoostingClassifier(**HGB)).fit(tr, y)

    # learned vs published weights (D2), on the raw 0-2/0-8/1-6/1-4 scales
    raw = LogisticRegression(**LR).fit(tr[DERM_ABCD], y)
    coef = dict(zip(DERM_ABCD, raw.coef_[0]))
    scaled = {k: round(v / coef["A"] * 1.3, 3) for k, v in coef.items()}
    d3 = fitted["D3"].named_steps["clf"].coef_[0]
    names3 = fitted["D3"].named_steps["pre"].get_feature_names_out()

    # holdout predictions saved for Stage 5 (NOT scored here)
    preds = ho[["image_id", "lesion_id", "dx", "y_melanoma", "TDS", "A", "B", "C", "D"]].copy()
    for rid, m in fitted.items():
        preds[f"p_{rid}"] = m.predict_proba(ho)[:, 1]
    preds.to_parquet(OUT / "ham10000_holdout_predictions.parquet", index=False)

    # PH2 external set: TDS uses OOF A/C; fitted models use features without context
    ph2 = pd.read_parquet(PREP / "ph2_features.parquet")
    ph2_in = ph2.assign(A=ph2.A_oof, C=ph2.C_oof)
    ph2_pred = ph2[["name", "clinical_dx"]].assign(TDS=ph2.TDS_oof)
    for rid in ("D2", "D3"):
        ph2_pred[f"p_{rid}"] = fitted[rid].predict_proba(ph2_in)[:, 1]
    ph2_pred.to_parquet(OUT / "ph2_external_predictions.parquet", index=False)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "chosen": chosen["run_id"], "tds_cut_sens85_train": cut,
                 "train_reference": tr[["image_id", "dx", "y_melanoma", "TDS"] + DERM_ABCD + DERM_CONT]},
                MODELS / "dermoscopic_models.joblib")
    return {"n_train": len(tr), "n_holdout": len(ho), "mel_train": int(y.sum()),
            "mel_holdout": int(ho.y_melanoma.sum()), "selected": chosen["run_id"],
            "selection_rule": "highest CV AUC among D2/D3/D3c; simplest within 0.02",
            "tds_recalibrated_cut_sens85_train": round(cut, 2),
            "published_cuts": [4.75, 5.45],
            "train_frac_TDS_gt_4.75": {"mel": round(float((s[y == 1] > 4.75).mean()), 3),
                                        "nv": round(float((s[y == 0] > 4.75).mean()), 3)},
            "learned_weights_D2_raw": {k: round(v, 3) for k, v in coef.items()},
            "learned_weights_D2_scaled_to_A=1.3": scaled,
            "published_weights": {"A": 1.3, "B": 0.1, "C": 0.5, "D": 0.5},
            "D3_standardised_coefficients": dict(sorted(zip(names3, d3.round(3)),
                                                        key=lambda kv: -abs(kv[1])))}


# ---------------------------------------------------------------- clinical

def clinical() -> dict:
    df = pd.read_parquet(PREP / "isic2024_features.parquet")
    groups = json.loads((REPORTS / "stage3" / "isic2024_feature_groups.json").read_text())
    core = [c for cols in groups["ABCD_CORE"].values() for c in cols]
    ugly, ctx_num, ctx_cat = groups["UGLY"], groups["CONTEXT_NUM"], groups["CONTEXT_CAT"]
    bench = [c for c in groups["BENCHMARK"] if c not in ("tbp_tile_type", "tbp_lv_location")]
    bench_cat = ctx_cat + ["tbp_tile_type", "tbp_lv_location"]
    df = df[df.y_melanoma.notna()].reset_index(drop=True)
    bench = [c for c in bench if pd.api.types.is_numeric_dtype(df[c])]  # Arrow strings slip past 'object'

    # holdout: 25% of patients, fixed before any fitting
    pat = df.groupby("patient_id").y_melanoma.max()
    p_tr, p_ho = train_test_split(pat.index.to_numpy(), test_size=0.25, stratify=pat.to_numpy(), random_state=SEED)
    df["split"] = np.where(df.patient_id.isin(p_ho), "holdout", "train")
    df[["isic_id", "patient_id", "split"]].to_csv(OUT / "isic2024_split.csv", index=False)  # local only (large)
    df.drop_duplicates("patient_id")[["patient_id", "split"]].sort_values("patient_id").to_csv(
        OUT / "isic2024_patient_split.csv", index=False)  # committed holdout definition
    tr_full, ho = df[df.split == "train"], df[df.split == "holdout"].reset_index(drop=True)

    # demo speed: keep all melanomas + 20,000 random benign crops for training
    rng = np.random.default_rng(SEED)
    neg = tr_full[tr_full.y_melanoma == 0]
    tr = pd.concat([tr_full[tr_full.y_melanoma == 1],
                    neg.iloc[rng.choice(len(neg), size=min(20000, len(neg)), replace=False)]]) \
        .reset_index(drop=True)
    y, g = tr.y_melanoma.astype(int).to_numpy(), tr.patient_id.to_numpy()

    a, p = fixed_score_cv(tr, y, g, np.zeros(len(tr)))
    log_run("C0", "clinical", "baseline: constant (prevalence)", [], {}, a, p, "train benign subsampled")
    a, p = fixed_score_cv(tr, y, g, tr.D_gt6mm.to_numpy().astype(float))
    log_run("C1", "clinical", "fixed rule: diameter > 6 mm", ["D_gt6mm"], {"cut_mm": 6}, a, p)
    # C1b ABCD count rule: A/B/C flagged above the 90th percentile of TRAIN benign
    # (no outcome tuning), D = >6 mm; score = number of positive criteria (0-4)
    flags = {}
    for crit, col in [("A", "tbp_lv_symm_2axis"), ("B", "tbp_lv_norm_border"), ("C", "tbp_lv_norm_color")]:
        flags[crit] = (col, float(tr.loc[y == 0, col].quantile(0.9)))
    def count_rule(d):
        return sum((d[c] > t).astype(int) for c, t in flags.values()) + d.D_gt6mm
    a, p = fixed_score_cv(tr, y, g, count_rule(tr).to_numpy().astype(float))
    log_run("C1b", "clinical", "fixed rule: ABCD count (A/B/C > benign P90, D > 6 mm)",
            list(flags) + ["D"], {k: v for k, v in flags.items()}, a, p)
    specs = {
        "C2": ("logistic regression on ABCD core", core, [], LogisticRegression(**LR)),
        "C3": ("logistic regression on ABCD core + ugly-duckling z-scores", core + ugly, [], LogisticRegression(**LR)),
        "C3c": ("logistic regression, ABCD + ugly duckling + age/sex/site", core + ugly + ctx_num, ctx_cat, LogisticRegression(**LR)),
        "C4": ("BENCHMARK gradient boosting, all TBP features + DNN + context", bench, bench_cat, HistGradientBoostingClassifier(**HGB)),
    }
    for rid, (name, num, cat, model) in specs.items():
        a, p, _ = cv_score(tr, y, g, num, cat, model)
        log_run(rid, "clinical", name, num + cat, model.get_params() if rid == "C4" else LR, a, p,
                "train benign subsampled to 20,000")
    cand = [r for r in RUNS if r["run_id"] in ("C2", "C3", "C3c")]
    best = max(cand, key=lambda r: r["cv_auc_mean"])
    chosen = min([r for r in cand if r["cv_auc_mean"] >= best["cv_auc_mean"] - 0.02],
                 key=lambda r: r["n_features"])
    fitted = {rid: make_pipe(num, cat, model).fit(tr, y) for rid, (_, num, cat, model) in specs.items()}
    preds = ho[["isic_id", "patient_id", "attribution", "strong_label", "iddx_3", "y_melanoma",
                "clin_size_long_diam_mm", "D_gt6mm"]].copy()
    preds["count_rule"] = count_rule(ho)
    for rid, m in fitted.items():
        preds[f"p_{rid}"] = m.predict_proba(ho)[:, 1]
    preds.to_parquet(OUT / "isic2024_holdout_predictions.parquet", index=False)
    c2 = fitted["C2"]
    coefs = dict(zip(c2.named_steps["pre"].get_feature_names_out(), c2.named_steps["clf"].coef_[0].round(3)))
    joblib.dump({"models": fitted, "chosen": chosen["run_id"], "count_rule_flags": flags},
                MODELS / "clinical_models.joblib")
    return {"n_train_full": len(tr_full), "n_train_used": len(tr), "mel_train": int(y.sum()),
            "n_holdout": len(ho), "mel_holdout": int(ho.y_melanoma.sum()),
            "patients_train": len(p_tr), "patients_holdout": len(p_ho),
            "selected": chosen["run_id"], "selection_rule": "highest CV AUC among C2/C3/C3c; simplest within 0.02",
            "count_rule_flags": flags, "C2_standardised_coefficients": dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1])))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"dermoscopic": dermoscopic(), "clinical": clinical()}
    pd.DataFrame(RUNS).to_csv(OUT / "run_log.csv", index=False)
    (OUT / "modeling_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
