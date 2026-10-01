"""Stage 3 preparation of the clinical ABCD(E) module (ISIC 2024 SLICE-3D).

Unit of analysis: one TBP lesion crop at the time of the total-body photo.
Writes ~/data/skin-cancer/prepared/isic2024_features.parquet (derived from licensed
data, kept outside the repo) and reports/stage3/isic2024_* (counts, dictionary,
checks).

Usage: uv run python -m skin_cancer.prep.isic2024
"""
import json
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from skin_cancer.config import DATA_DIR, ISIC2024, REPORTS

OUT = REPORTS / "stage3"
PREP = DATA_DIR / "prepared"

# Canfield Lesion Visualizer features mapped to ABCD (Kurtansky et al. 2024, Table 1).
# CORE = one or two per criterion for the interpretable model (low collinearity).
ABCD_CORE = {
    "A": ["tbp_lv_symm_2axis", "tbp_lv_radial_color_std_max"],
    "B": ["tbp_lv_norm_border", "tbp_lv_area_perim_ratio"],
    "C": ["tbp_lv_norm_color", "tbp_lv_deltaLBnorm"],
    "D": ["clin_size_long_diam_mm"],
}
CORE = [c for cols in ABCD_CORE.values() for c in cols]
UGLY = [f"{c}_pz" for c in CORE]  # patient-relative z-scores ("ugly duckling")
CONTEXT_NUM = ["age_approx"]
CONTEXT_CAT = ["sex", "anatom_site_general"]
DNN = ["tbp_lv_nevi_confidence", "tbp_lv_dnn_lesion_confidence"]
EXCLUDED = {
    "lesion_id": "LEAK: present on 100% of malignant vs 5.4% of benign crops (02 §3)",
    "iddx_full": "LEAK: diagnosis", "iddx_1": "LEAK: diagnosis", "iddx_2": "LEAK: diagnosis",
    "iddx_3": "LEAK: diagnosis", "iddx_4": "LEAK: diagnosis", "iddx_5": "LEAK: diagnosis",
    "mel_mitotic_index": "LEAK: histopathology", "mel_thick_mm": "LEAK: histopathology",
    "attribution": "SHORTCUT: institution, not known at bedside; robustness strata only",
    "copyright_license": "SHORTCUT: proxy for institution",
    "image_type": "constant",
    "isic_id": "identifier", "patient_id": "grouping key only",
}
TARGETS = ["y_melanoma", "y_malignant"]

DEFINITIONS = {
    "tbp_lv_symm_2axis": ("A", "Border asymmetry about the axis perpendicular to the most symmetric axis, 0-10 (Lesion Visualizer)"),
    "tbp_lv_radial_color_std_max": ("A", "Colour asymmetry: max radial std of colour (Lesion Visualizer)"),
    "tbp_lv_norm_border": ("B", "Border irregularity 0-10 (mean of jaggedness and asymmetry)"),
    "tbp_lv_area_perim_ratio": ("B", "Border jaggedness: perimeter^2/area"),
    "tbp_lv_norm_color": ("C", "Colour variation 0-10"),
    "tbp_lv_deltaLBnorm": ("C", "Contrast between lesion and surrounding skin (L*B* normalised)"),
    "clin_size_long_diam_mm": ("D", "Longest diameter in mm (calibrated 3D TBP)"),
    "age_approx": ("context", "Patient age, 5-year bins"),
    "sex": ("context", "Patient sex"),
    "anatom_site_general": ("context", "Anatomic site (5 groups)"),
}


def load() -> pd.DataFrame:
    meta = pd.read_csv(zipfile.ZipFile(ISIC2024 / "ISIC_2024_Training_Input.zip")
                       .open("ISIC_2024_Training_Input/metadata.csv"), low_memory=False)
    gt = pd.read_csv(ISIC2024 / "ISIC_2024_Training_GroundTruth.csv")
    sup = pd.read_csv(ISIC2024 / "ISIC_2024_Training_Supplement.csv", low_memory=False)
    dup = [c for c in sup.columns if c in meta.columns and c != "isic_id"]
    df = meta.merge(gt, on="isic_id", validate="one_to_one") \
             .merge(sup.drop(columns=dup), on="isic_id", validate="one_to_one")
    return df


def patient_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """z-score of each feature among the same patient's lesions (labels unused)."""
    g = df.groupby("patient_id")[cols]
    mu, sd = g.transform("mean"), g.transform("std")
    z = (df[cols] - mu) / sd.where(sd > 0)
    return z.fillna(0.0).add_suffix("_pz")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    counts = []
    df = load()
    counts.append(("loaded + joined (metadata x groundtruth x supplement)", len(df), int(df.malignant.sum())))

    # patient-relative features use ALL of a patient's crops, before any row filtering
    df["n_lesions_patient"] = df.groupby("patient_id").isic_id.transform("size")
    df = pd.concat([df, patient_z(df, CORE)], axis=1)

    df = df[df.iddx_1 != "Indeterminate"].copy()
    counts.append(("drop Indeterminate (ambiguous histology)", len(df), int(df.malignant.sum())))

    is_mel = df.iddx_3.fillna("").str.contains("Melanoma")
    df["y_malignant"] = df.malignant.astype(int)
    df["y_melanoma"] = np.where(is_mel, 1.0, np.where(df.malignant == 0, 0.0, np.nan))
    df["strong_label"] = ((df.malignant == 1) | df.iddx_2.notna()).astype(int)  # eval only
    counts.append(("of which usable for melanoma target (BCC/SCC set to NaN)",
                   int(df.y_melanoma.notna().sum()), int(np.nansum(df.y_melanoma))))

    df["D_gt6mm"] = (df.clin_size_long_diam_mm > 6).astype(int)  # published clinical rule
    for c in CONTEXT_NUM + CONTEXT_CAT:
        df[f"{c}_missing"] = df[c].isna().astype(int)        # imputation happens in Stage 4

    tbp_all = [c for c in df.columns if c.startswith("tbp_lv_") and c not in DNN
               and df[c].dtype != object]
    benchmark = sorted(set(tbp_all + ["clin_size_long_diam_mm", "n_lesions_patient"]
                           + UGLY + CONTEXT_NUM + DNN))
    keep = (["isic_id", "patient_id", "attribution", "strong_label", "iddx_3", "tbp_tile_type",
             "tbp_lv_location"] + TARGETS + CORE + UGLY + ["D_gt6mm"] + CONTEXT_NUM + CONTEXT_CAT
            + [f"{c}_missing" for c in CONTEXT_NUM + CONTEXT_CAT] + benchmark)
    keep = list(dict.fromkeys(keep))
    out = df[keep]
    out.to_parquet(PREP / "isic2024_features.parquet", index=False)

    groups = {"ABCD_CORE": ABCD_CORE, "UGLY": UGLY, "CONTEXT_NUM": CONTEXT_NUM,
              "CONTEXT_CAT": CONTEXT_CAT, "BENCHMARK": benchmark + ["tbp_tile_type", "tbp_lv_location"],
              "DNN_benchmark_only": DNN, "EXCLUDED": EXCLUDED, "TARGETS": TARGETS,
              "EVAL_ONLY": ["strong_label", "attribution", "iddx_3"]}
    (OUT / "isic2024_feature_groups.json").write_text(json.dumps(groups, indent=2))
    pd.DataFrame(counts, columns=["step", "rows", "malignant"]).to_csv(
        OUT / "isic2024_row_counts.csv", index=False)

    # data dictionary for the modelling features
    rows = []
    for c in CORE + UGLY + ["D_gt6mm"] + CONTEXT_NUM + CONTEXT_CAT:
        base = c[:-3] if c.endswith("_pz") else c
        crit, desc = DEFINITIONS.get(base, ("rule", "Published clinical rule: long diameter > 6 mm")
                                     if c == "D_gt6mm" else ("?", ""))
        if c.endswith("_pz"):
            desc = f"Patient-relative z-score of {base} among the patient's own lesions ('ugly duckling')"
        rows.append({"feature": c, "criterion": crit, "definition": desc,
                     "source": "ISIC 2024 metadata.csv (Canfield Lesion Visualizer)" if base.startswith(("tbp", "clin")) else "ISIC 2024 metadata.csv",
                     "type": str(out[c].dtype), "missing_frac": round(float(out[c].isna().mean()), 4),
                     "missing_treatment": "indicator + median impute inside CV folds" if c in CONTEXT_NUM
                     else ("indicator + 'missing' category" if c in CONTEXT_CAT else "none needed"),
                     "decision_time": "Y: produced by the TBP scan / visible or askable at the visit"})
    pd.DataFrame(rows).to_csv(OUT / "isic2024_data_dictionary.csv", index=False)

    # investigation: is the reversed asymmetry signal a size artefact?
    mel = out[out.y_melanoma.notna()].copy()
    rho = {c: round(float(mel[c].corr(mel.clin_size_long_diam_mm, method="spearman")), 3)
           for c in ["tbp_lv_symm_2axis", "tbp_lv_norm_border", "tbp_lv_area_perim_ratio",
                     "tbp_lv_radial_color_std_max", "tbp_lv_norm_color"]}
    mel["size_q"] = pd.qcut(mel.clin_size_long_diam_mm, 5, labels=False)
    strata = []
    for q, g in mel.groupby("size_q"):
        n_mel = int(g.y_melanoma.sum())
        row = {"size_quintile": int(q), "diam_min": round(g.clin_size_long_diam_mm.min(), 2),
               "diam_max": round(g.clin_size_long_diam_mm.max(), 2), "n": len(g), "n_melanoma": n_mel}
        if n_mel >= 5:
            for c in ["tbp_lv_symm_2axis", "tbp_lv_norm_border", "tbp_lv_norm_color"]:
                row[f"auc_{c}"] = round(float(roc_auc_score(g.y_melanoma, g[c])), 3)
        strata.append(row)
    pd.DataFrame(strata).to_csv(OUT / "isic2024_asymmetry_size_check.csv", index=False)
    (OUT / "isic2024_size_correlation.json").write_text(json.dumps(
        {"spearman_with_long_diameter": rho}, indent=2))
    print(pd.DataFrame(counts).to_string(), "\n", json.dumps(rho), "\n", pd.DataFrame(strata).to_string())


if __name__ == "__main__":
    main()
