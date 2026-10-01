"""Stage 2 profiling of the freely available dermoscopic sources (HAM10000,
ISIC 2018 Task 2 attribute masks). Writes reports/stage2/dermoscopy_profile.json.

Usage: uv run python -m skin_cancer.data.profile_dermoscopy
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from skin_cancer.config import DATA_DIR, REPORTS

OUT = REPORTS / "stage2"
HAM = DATA_DIR / "ham10000"
T2 = DATA_DIR / "isic2018_task2"


def profile_ham() -> dict:
    m = pd.read_csv(HAM / "HAM10000_metadata.tab", sep=None, engine="python")
    lesions = m.drop_duplicates("lesion_id")
    with zipfile.ZipFile(HAM / "HAM10000_segmentations_lesion_tschandl.zip") as z:
        n_masks = sum(n.endswith(".png") for n in z.namelist())
    return {
        "n_images": int(len(m)),
        "n_lesions": int(m.lesion_id.nunique()),
        "images_per_lesion_max": int(m.groupby("lesion_id").size().max()),
        "dx_images": m.dx.value_counts().to_dict(),
        "dx_lesions": lesions.dx.value_counts().to_dict(),
        "dx_type_images": m.dx_type.value_counts().to_dict(),
        "dx_type_by_dx_lesions": pd.crosstab(lesions.dx, lesions.dx_type).to_dict("index"),
        "melanocytic_mel_vs_nv_lesions": {
            "mel": int((lesions.dx == "mel").sum()), "nv": int((lesions.dx == "nv").sum())},
        "source_dataset": m.dataset.value_counts().to_dict(),
        "missing": m.isna().sum()[lambda s: s > 0].to_dict(),
        "age": m.age.describe().round(1).to_dict(),
        "n_segmentation_masks": n_masks,
        "patient_id_available": False,
    }


def profile_task2() -> dict | None:
    z = T2 / "ISIC2018_Task2_Training_GroundTruth_v3.zip"
    if not z.exists():
        return None
    counts, n_imgs = {}, set()
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith(".png"):
                continue
            stem = n.rsplit("/", 1)[-1][:-4]          # ISIC_xxx_attribute_<name>
            img, attr = stem.split("_attribute_")
            n_imgs.add(img)
            nonempty = np.asarray(Image.open(io.BytesIO(zf.read(n)))).max() > 0
            c = counts.setdefault(attr, [0, 0])
            c[0] += int(nonempty)
            c[1] += 1
    return {"n_images": len(n_imgs),
            "attribute_nonempty_images": {k: v[0] for k, v in counts.items()},
            "attribute_masks_total": {k: v[1] for k, v in counts.items()}}


PH2 = DATA_DIR / "ph2"
PH2_COLS = ["name", "histology", "clinical_dx", "asymmetry", "pigment_network",
            "dots_globules", "streaks", "regression", "blue_white_veil", "colors"]


def load_ph2() -> pd.DataFrame:
    """Parse the '||'-delimited PH2_dataset.txt (unofficial GitHub mirror)."""
    rows = []
    for line in (PH2 / "PH2_dataset.txt").read_text().splitlines():
        if not line.startswith("|| IMD"):
            continue
        cells = [c.strip() for c in line.replace("||", "|").strip("|").split("|")]
        rows.append(dict(zip(PH2_COLS, cells)))
    df = pd.DataFrame(rows)
    df["clinical_dx"] = df.clinical_dx.astype(int)
    df["asymmetry"] = df.asymmetry.astype(int)
    df["n_colors"] = df.colors.str.split().str.len()
    return df


def profile_ph2() -> dict | None:
    if not (PH2 / "PH2_dataset.txt").exists():
        return None
    from sklearn.metrics import roc_auc_score
    df = load_ph2()
    with zipfile.ZipFile(PH2 / "PH2.zip") as z:
        imgs = {n.rsplit("/", 1)[-1][:-4] for n in z.namelist()
                if n.startswith("ph2_dataset/trainx/") and n.endswith(".bmp")}
        masks = {n.rsplit("/", 1)[-1][:-4] for n in z.namelist()
                 if n.startswith("ph2_dataset/trainy/") and n.endswith(".bmp")}
    y = (df.clinical_dx == 2).astype(int)
    return {
        "n_cases": int(len(df)),
        "clinical_dx_counts": df.clinical_dx.value_counts().sort_index().to_dict(),
        "histology_present_by_dx": df.assign(h=df.histology != "").groupby("clinical_dx")
                                     .h.sum().astype(int).to_dict(),
        "images_in_zip": len(imgs), "masks_in_zip": len(masks),
        "annotation_ids_missing_image": sorted(set(df.name) - imgs),
        "mask_names_sample": sorted(masks)[:3],
        "asymmetry_by_dx": pd.crosstab(df.clinical_dx, df.asymmetry).to_dict("index"),
        "n_colors_by_dx": df.groupby("clinical_dx").n_colors.describe().round(2)
                            .to_dict("index"),
        "structures_by_dx": {c: pd.crosstab(df.clinical_dx, df[c]).to_dict("index")
                             for c in ["pigment_network", "dots_globules", "streaks",
                                       "regression", "blue_white_veil"]},
        "expert_auc_melanoma_vs_nevi": {
            "asymmetry": round(float(roc_auc_score(y, df.asymmetry)), 4),
            "n_colors": round(float(roc_auc_score(y, df.n_colors)), 4)},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = {"ham10000": profile_ham(), "isic2018_task2": profile_task2(), "ph2": profile_ph2()}
    (OUT / "dermoscopy_profile.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
