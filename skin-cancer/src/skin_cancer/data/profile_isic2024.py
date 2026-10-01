"""Stage 2 profiling of ISIC 2024 SLICE-3D. All numbers cited in
crisp/02_data_understanding.md come from the files this script writes to
reports/stage2/.

Usage: uv run python -m skin_cancer.data.profile_isic2024
"""
import json
import zipfile

import pandas as pd

from skin_cancer.config import ISIC2024, REPORTS

OUT = REPORTS / "stage2"

# Candidate ABCD(E) proxies among the TBP Lesion-Visualizer features.
ABCD_PROXIES = {
    "A": ["tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle", "tbp_lv_eccentricity"],
    "B": ["tbp_lv_norm_border", "tbp_lv_perimeterMM", "tbp_lv_area_perim_ratio"],
    "C": ["tbp_lv_norm_color", "tbp_lv_color_std_mean", "tbp_lv_radial_color_std_max",
          "tbp_lv_deltaLBnorm"],
    "D": ["clin_size_long_diam_mm", "tbp_lv_areaMM2", "tbp_lv_minorAxisMM"],
}
# Fields derived from the diagnosis -> never usable as features.
LEAKY = ["iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
         "mel_mitotic_index", "mel_thick_mm", "target", "malignant", "lesion_id"]


def load_metadata_from_zip() -> pd.DataFrame | None:
    zpath = ISIC2024 / "ISIC_2024_Training_Input.zip"
    if not zpath.exists():
        return None
    with zipfile.ZipFile(zpath) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        (OUT / "zip_contents.json").write_text(json.dumps(
            {"n_files": len(z.namelist()), "csv_files": csvs,
             "n_jpg": sum(n.lower().endswith(".jpg") for n in z.namelist())}, indent=2))
        if not csvs:
            return None
        with z.open(csvs[0]) as f:
            return pd.read_csv(f, low_memory=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(ISIC2024 / "ISIC_2024_Training_GroundTruth.csv")
    sup = pd.read_csv(ISIC2024 / "ISIC_2024_Training_Supplement.csv", low_memory=False)
    df = gt.merge(sup, on="isic_id", how="outer", validate="one_to_one")

    meta = load_metadata_from_zip()
    if meta is not None:
        dup = [c for c in meta.columns if c in df.columns and c != "isic_id"]
        df = df.merge(meta.drop(columns=dup), on="isic_id", how="left",
                      validate="one_to_one")

    s = {
        "n_images": int(len(df)),
        "n_malignant": int(df["malignant"].sum()),
        "base_rate_malignant": float(df["malignant"].mean()),
        "label_missing": int(df["malignant"].isna().sum()),
        "n_lesion_id_present": int(df["lesion_id"].notna().sum()),
        "malignant_with_lesion_id": int(df.loc[df.malignant == 1, "lesion_id"].notna().sum()),
        "licence_counts": df["copyright_license"].value_counts().to_dict(),
        "attribution_counts": df["attribution"].value_counts().to_dict(),
        "malignant_by_licence": df.groupby("copyright_license")["malignant"]
                                  .agg(["sum", "count"]).to_dict("index"),
        "iddx_1_counts": df["iddx_1"].value_counts().to_dict(),
        "iddx_3_malignant": df.loc[df.malignant == 1, "iddx_3"].value_counts().to_dict(),
        "has_zip_metadata": meta is not None,
    }
    if meta is not None:
        s["n_columns"] = int(df.shape[1])
        s["columns"] = list(df.columns)
        s["n_patients"] = int(df["patient_id"].nunique())
        per_pat = df.groupby("patient_id")["malignant"].agg(["count", "sum"])
        s["images_per_patient"] = per_pat["count"].describe().round(1).to_dict()
        s["patients_with_malignant"] = int((per_pat["sum"] > 0).sum())
        s["malignant_per_positive_patient"] = per_pat.loc[per_pat["sum"] > 0, "sum"] \
            .describe().round(2).to_dict()
        for col in ["sex", "anatom_site_general", "tbp_tile_type", "image_type",
                    "tbp_lv_location_simple", "attribution"]:
            if col in df:
                s[f"malignant_rate_by_{col}"] = df.groupby(col, dropna=False)["malignant"] \
                    .agg(["sum", "count", "mean"]).round(5).reset_index() \
                    .astype({col: str}).to_dict("records")
        if "age_approx" in df:
            s["age_approx"] = df["age_approx"].describe().round(1).to_dict()

        miss = df.isna().mean().sort_values(ascending=False)
        miss[miss > 0].round(4).to_csv(OUT / "isic2024_missingness.csv", header=["frac_missing"])

        # Univariate signal of each ABCD proxy: AUC of the raw feature, with direction.
        from sklearn.metrics import roc_auc_score
        rows = []
        for crit, cols in ABCD_PROXIES.items():
            for c in cols:
                if c not in df:
                    rows.append({"criterion": crit, "feature": c, "present": False})
                    continue
                m = df[[c, "malignant"]].dropna()
                auc = roc_auc_score(m["malignant"], m[c])
                rows.append({
                    "criterion": crit, "feature": c, "present": True,
                    "missing_frac": round(float(df[c].isna().mean()), 4),
                    "auc_raw": round(auc, 4), "auc_best_dir": round(max(auc, 1 - auc), 4),
                    "median_benign": round(float(m.loc[m.malignant == 0, c].median()), 3),
                    "median_malignant": round(float(m.loc[m.malignant == 1, c].median()), 3),
                })
        pd.DataFrame(rows).to_csv(OUT / "isic2024_abcd_proxy_auc.csv", index=False)

        # ABCD is a melanoma rule: repeat the univariate check for melanoma vs benign
        # (BCC/SCC removed), raw and patient-relative ("ugly duckling": z-score of the
        # feature among the same patient's lesions, available at decision time in TBP).
        df["is_melanoma"] = df["iddx_3"].fillna("").str.contains("Melanoma").astype(int)
        mel = df[(df.malignant == 0) | (df.is_melanoma == 1)].copy()
        s["melanoma_subset"] = {"n": int(len(mel)), "n_melanoma": int(mel.is_melanoma.sum()),
                                "base_rate": float(mel.is_melanoma.mean())}
        rows = []
        for crit, cols in ABCD_PROXIES.items():
            for c in cols:
                g = mel.groupby("patient_id")[c]
                z = (mel[c] - g.transform("mean")) / g.transform("std").replace(0, 1)
                for kind, x in [("raw", mel[c]), ("patient_z", z)]:
                    m = pd.DataFrame({"x": x, "y": mel.is_melanoma}).dropna()
                    auc = roc_auc_score(m.y, m.x)
                    rows.append({"criterion": crit, "feature": c, "kind": kind,
                                 "auc_raw": round(auc, 4),
                                 "auc_best_dir": round(max(auc, 1 - auc), 4)})
        pd.DataFrame(rows).to_csv(OUT / "isic2024_abcd_proxy_auc_melanoma.csv", index=False)
        if "clin_size_long_diam_mm" in mel:
            d6, y = mel["clin_size_long_diam_mm"] > 6, mel.is_melanoma == 1
            s["melanoma_diameter_gt6mm"] = {
                "sensitivity": float((d6 & y).sum() / y.sum()),
                "specificity": float((~d6 & ~y).sum() / (~y).sum())}
            s["melanoma_long_diam_mm"] = mel.loc[y, "clin_size_long_diam_mm"] \
                .describe().round(2).to_dict()

        # Label quality: how was the benign label assigned?
        s["iddx_full_top_benign"] = df.loc[df.malignant == 0, "iddx_full"] \
            .value_counts().head(15).to_dict()
        s["indeterminate_iddx_full"] = df.loc[df.iddx_1 == "Indeterminate", "iddx_full"] \
            .value_counts().head(10).to_dict()
        s["lesion_id_by_label"] = pd.crosstab(df.lesion_id.notna(), df.malignant) \
            .rename(index=str, columns=str).to_dict()
        s["benign_with_iddx2"] = int(df.loc[df.malignant == 0, "iddx_2"].notna().sum())
        s["dnn_confidence_auc"] = {
            c: round(float(roc_auc_score(df.malignant, df[c].fillna(df[c].median()))), 4)
            for c in ["tbp_lv_nevi_confidence", "tbp_lv_dnn_lesion_confidence"]}

        # Clinical "D" rule: long diameter > 6 mm.
        if "clin_size_long_diam_mm" in df:
            d6 = df["clin_size_long_diam_mm"] > 6
            mal = df["malignant"] == 1
            s["diameter_gt6mm"] = {
                "frac_lesions_gt6": float(d6.mean()),
                "sensitivity": float((d6 & mal).sum() / mal.sum()),
                "specificity": float((~d6 & ~mal).sum() / (~mal).sum()),
            }
        df.describe(include="number").T.round(4).to_csv(OUT / "isic2024_numeric_summary.csv")

    (OUT / "isic2024_profile.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps({k: v for k, v in s.items() if k != "columns"}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
