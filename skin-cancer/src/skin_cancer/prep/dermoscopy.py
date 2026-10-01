"""Stage 3 preparation of the dermoscopic module (PH2, ISIC 2018 Task 2, HAM10000).

Steps (each writes outputs; numbers in crisp/03 come from reports/stage3/):
  ph2     extract raw ABCD features for PH2; calibrate A and C thresholds against
          the PH2 expert labels with 5-fold out-of-fold (OOF) scoring
  task2   extract D-structure features on ISIC 2018 Task 2 (minus HAM10000 overlap)
          and calibrate D thresholds against the expert attribute masks
  ham     extract features for every HAM10000 image with the calibrated thresholds
  ph2val  validate D on PH2 structure labels and write PH2 OOF TDS (external set)

  finalize  rescore HAM10000 with final thresholds, add targets/cohorts, dictionary

Usage: uv run python -m skin_cancer.prep.dermoscopy [ph2 task2 ham ph2val finalize]
"""
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import color as skcolor
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from skin_cancer.config import DATA_DIR, REPORTS
from skin_cancer.data.profile_dermoscopy import load_ph2
from skin_cancer.prep.abcd_image import (COLOR_NAMES, Thresholds, clean_mask, extract,
                                         remove_hair, resize_pair)

OUT = REPORTS / "stage3"
PREP = DATA_DIR / "prepared"
THRESH_FILE = OUT / "dermoscopy_thresholds.json"
PH2_COLOR_CODE = {"1": "white", "2": "red", "3": "light_brown", "4": "dark_brown",
                  "5": "blue_gray", "6": "black"}


# ------------------------------------------------------------------ io helpers

def _ph2_pair(z: zipfile.ZipFile, name: str):
    rgb = np.array(Image.open(io.BytesIO(z.read(f"ph2_dataset/trainx/{name}.bmp"))).convert("RGB"))
    m = np.array(Image.open(io.BytesIO(z.read(f"ph2_dataset/trainy/{name}_lesion.bmp")))
                 .convert("L")) > 127
    return rgb, m


def _safe_extract(args):
    key, rgb, mask, t = args
    try:
        return {"key": key, **extract(rgb, mask, t).flat(), "error": None}
    except Exception as e:  # recorded, never silently dropped
        return {"key": key, "error": repr(e)}


def _run(jobs, workers=8):
    with ProcessPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(_safe_extract, jobs, chunksize=8)))


def load_thresholds() -> Thresholds:
    if THRESH_FILE.exists():
        return Thresholds(**json.loads(THRESH_FILE.read_text())["thresholds"])
    return Thresholds()


def save_thresholds(t: Thresholds, provenance: dict) -> None:
    THRESH_FILE.write_text(json.dumps({"thresholds": asdict(t), "provenance": provenance},
                                      indent=2))


# ------------------------------------------------------------------------- PH2

def ph2_lab_reference(df: pd.DataFrame, z: zipfile.ZipFile) -> dict:
    """Median skin and lesion Lab in PH2 (justifies CANON_SKIN_LAB and brown anchors)."""
    skins, rows = [], []
    for name, cols in zip(df.name, df.colors):
        rgb, m = _ph2_pair(z, name)
        rgb, m = resize_pair(rgb, m)
        m = clean_mask(m)
        lab = skcolor.rgb2lab(remove_hair(rgb))
        d = ndi.distance_transform_edt(~m)
        skins.append(np.median(lab[(d > 5) & (d < 25)], 0))
        rows.append((cols.strip(), np.median(lab[m], 0)))
    ref = {"skin_median_lab": np.median(skins, 0).round(1).tolist()}
    for key, label in [("3", "light_brown_only"), ("4", "dark_brown_only")]:
        v = [l for c, l in rows if c == key]
        ref[f"lesion_median_lab_{label}"] = np.median(v, 0).round(1).tolist()
        ref[f"n_{label}"] = len(v)
    return ref


def a_score(f: pd.DataFrame, a_shape: float, a_color: float) -> np.ndarray:
    ax1 = (f.a_shape_ax1 > a_shape) | (f.a_color_ax1 > a_color)
    ax2 = (f.a_shape_ax2 > a_shape) | (f.a_color_ax2 > a_color)
    return (ax1.astype(int) + ax2.astype(int)).to_numpy()


def c_score(f: pd.DataFrame, c_min) -> np.ndarray:
    """c_min: scalar or per-colour sequence of presence thresholds."""
    fr = f[[f"c_frac_{n}" for n in COLOR_NAMES]].to_numpy()
    return np.maximum(1, (fr >= np.asarray(c_min)).sum(1))


A_GRID = [(s, c) for s in np.round(np.arange(0.05, 0.61, 0.025), 3)
          for c in np.round(np.arange(2, 30.1, 1.0), 1)]
C_GRID = np.round(np.arange(0.01, 0.301, 0.01), 3)


def _fit_a(f, y):
    return max(A_GRID, key=lambda p: cohen_kappa_score(y, a_score(f, *p), weights="quadratic"))


def _fit_c(f, y=None):
    """One presence threshold per colour, each fitted to that colour's expert label."""
    return [float(max(C_GRID, key=lambda c: cohen_kappa_score(
        f[f"expert_has_{n}"], f[f"c_frac_{n}"] >= c))) for n in COLOR_NAMES]


def step_ph2() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    df = load_ph2()
    z = zipfile.ZipFile(DATA_DIR / "ph2" / "PH2.zip")
    (OUT / "ph2_lab_reference.json").write_text(json.dumps(ph2_lab_reference(df, z), indent=2))

    t0 = Thresholds()
    feats = _run([(n, *_ph2_pair(z, n), t0) for n in df.name])
    assert feats.error.isna().all(), feats[feats.error.notna()]
    f = df.merge(feats.rename(columns={"key": "name"}), on="name", validate="one_to_one")
    f["expert_C"] = f.colors.str.split().str.len()
    for code, n in PH2_COLOR_CODE.items():
        f[f"expert_has_{n}"] = f.colors.str.split().apply(lambda c: code in c).astype(int)

    # 5-fold OOF calibration of A and C (stratified on the diagnosis)
    f["fold"] = -1
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (_, te) in enumerate(skf.split(f, f.clinical_dx)):
        f.loc[f.index[te], "fold"] = k
    f["A_oof"], f["C_oof"] = 0, 0
    fold_params = []
    for k in range(5):
        tr, te = f.fold != k, f.fold == k
        pa = _fit_a(f[tr], f.asymmetry[tr])
        pc = _fit_c(f[tr], f.expert_C[tr])
        f.loc[te, "A_oof"] = a_score(f[te], *pa)
        f.loc[te, "C_oof"] = c_score(f[te], pc)
        fold_params.append({"fold": k, "a_shape": pa[0], "a_color": pa[1], "c_min_frac_per": pc})
    pa_all, pc_all = _fit_a(f, f.asymmetry), _fit_c(f, f.expert_C)

    # comparison variant: one global presence threshold fitted to the count kappa
    c_glob = np.zeros(len(f), int)
    for k in range(5):
        tr, te = (f.fold != k).to_numpy(), (f.fold == k).to_numpy()
        g = max(C_GRID, key=lambda c: cohen_kappa_score(f.expert_C[tr], c_score(f[tr], c),
                                                        weights="quadratic"))
        c_glob[te] = c_score(f[te], g)
    c_global_variant = {
        "kappa_quadratic_oof": round(float(cohen_kappa_score(f.expert_C, c_glob, weights="quadratic")), 3),
        "exact_agreement_oof": round(float((f.expert_C == c_glob).mean()), 3),
        "within_one_oof": round(float((np.abs(f.expert_C - c_glob) <= 1).mean()), 3)}

    c_min_oof = {r["fold"]: r["c_min_frac_per"] for r in fold_params}
    per_colour = {}
    for j, n in enumerate(COLOR_NAMES):
        pred = np.array([f.loc[i, f"c_frac_{n}"] >= c_min_oof[f.loc[i, "fold"]][j] for i in f.index])
        per_colour[n] = {
            "expert_prevalence": int(f[f"expert_has_{n}"].sum()),
            "pred_prevalence_oof": int(pred.sum()),
            "kappa_oof": round(float(cohen_kappa_score(f[f"expert_has_{n}"], pred)), 3),
            "auc_frac": round(float(roc_auc_score(f[f"expert_has_{n}"], f[f"c_frac_{n}"])), 3)
            if 0 < f[f"expert_has_{n}"].sum() < len(f) else None,
        }
    val = {
        "n": int(len(f)),
        "A": {"kappa_quadratic_oof": round(float(cohen_kappa_score(f.asymmetry, f.A_oof, weights="quadratic")), 3),
              "kappa_unweighted_oof": round(float(cohen_kappa_score(f.asymmetry, f.A_oof)), 3),
              "kappa_binary_oof(0 vs 1-2)": round(float(cohen_kappa_score(f.asymmetry > 0, f.A_oof > 0)), 3),
              "confusion_oof": pd.crosstab(f.asymmetry, f.A_oof).to_dict("index"),
              "fold_params": fold_params, "final_params": {"a_shape": pa_all[0], "a_color": pa_all[1]}},
        "C": {"kappa_quadratic_oof": round(float(cohen_kappa_score(f.expert_C, f.C_oof, weights="quadratic")), 3),
              "exact_agreement_oof": round(float((f.expert_C == f.C_oof).mean()), 3),
              "within_one_oof": round(float(((f.expert_C - f.C_oof).abs() <= 1).mean()), 3),
              "confusion_oof": pd.crosstab(f.expert_C, f.C_oof).to_dict("index"),
              "per_colour": per_colour, "final_params": {"c_min_frac_per": pc_all}},
        "C_global_threshold_variant(not used)": c_global_variant,
        "B_distribution_default": f.B.value_counts().sort_index().to_dict(),
        "B_by_dx": f.groupby("clinical_dx").B.mean().round(2).to_dict(),
    }
    (OUT / "ph2_abcd_validation.json").write_text(json.dumps(val, indent=2, default=str))
    f.drop(columns=[c for c in f.columns if c in ("error",)]).to_parquet(PREP / "ph2_features_raw.parquet")

    t = replace(load_thresholds(), a_shape=float(pa_all[0]), a_color=float(pa_all[1]),
                c_min_frac_per=pc_all)
    save_thresholds(t, {"A,C": "PH2 expert labels, grid search on all 200 (OOF kappa in ph2_abcd_validation.json)"})
    print(json.dumps({k: v for k, v in val.items() if k in ("A", "C")}, indent=1, default=str)[:3000])


# ----------------------------------------------------------------- Task 2 (D)

T2_ATTRS = {"pigment_network": "d_net_score", "globules": None, "streaks": "d_streak_score"}


_ZIPS: dict = {}


def _zip(path) -> zipfile.ZipFile:
    if path not in _ZIPS:            # one handle per worker process
        _ZIPS[path] = zipfile.ZipFile(path)
    return _ZIPS[path]


def _task2_job(args):
    """Worker: read + downsize a full-resolution ISIC 2018 image and its Task 1 mask."""
    iid, img_zip, img_name, mask_zip, mask_name, t = args
    import cv2
    rgb = np.array(Image.open(io.BytesIO(_zip(img_zip).read(img_name))).convert("RGB"))
    m = np.array(Image.open(io.BytesIO(_zip(mask_zip).read(mask_name))).convert("L")) > 127
    s = 1024 / max(rgb.shape[:2])
    if s < 1:
        size = (round(rgb.shape[1] * s), round(rgb.shape[0] * s))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return _safe_extract((iid, rgb, m, t))


def step_task2() -> None:
    t2 = DATA_DIR / "isic2018_task2"
    img_zip = t2 / "ISIC2018_Task1-2_Training_Input.zip"
    mask_zip = t2 / "ISIC2018_Task1_Training_GroundTruth.zip"
    zi, zm = zipfile.ZipFile(img_zip), zipfile.ZipFile(mask_zip)
    zg = zipfile.ZipFile(t2 / "ISIC2018_Task2_Training_GroundTruth_v3.zip")
    ham_ids = set(pd.read_csv(DATA_DIR / "ham10000" / "HAM10000_metadata.tab", sep=None,
                              engine="python").image_id)
    imgs = {n.rsplit("/", 1)[-1][:-4]: n for n in zi.namelist() if n.endswith(".jpg")}
    masks = {n.rsplit("/", 1)[-1].replace("_segmentation.png", ""): n
             for n in zm.namelist() if n.endswith(".png")}
    gt = {}
    for n in zg.namelist():
        if n.endswith(".png"):
            stem = n.rsplit("/", 1)[-1][:-4]
            iid, attr = stem.split("_attribute_")
            gt.setdefault(iid, {})[attr] = n
    ids = sorted(set(imgs) & set(gt) & set(masks))
    overlap = [i for i in ids if i in ham_ids]
    ids = [i for i in ids if i not in ham_ids]

    t = load_thresholds()
    jobs = [(i, str(img_zip), imgs[i], str(mask_zip), masks[i], t) for i in ids]
    with ProcessPoolExecutor(8) as ex:
        feats = pd.DataFrame(list(ex.map(_task2_job, jobs, chunksize=8)))
    labels = []
    for iid in ids:
        row = {"key": iid}
        for attr, n in gt[iid].items():
            row[f"gt_{attr}"] = int(np.asarray(Image.open(io.BytesIO(zg.read(n)))).max() > 0)
        labels.append(row)
    f = feats.merge(pd.DataFrame(labels), on="key", validate="one_to_one")
    f.to_parquet(PREP / "task2_features.parquet")
    ok = f.error.isna()

    # calibrate D thresholds by Youden's J on the continuous scores
    def youden(score, y, grid, ge=False):
        """Threshold maximising sensitivity + specificity; ge=True for counts (>=)."""
        def j(g):
            pos = score >= g if ge else score > g
            return (pos & (y == 1)).sum() / max((y == 1).sum(), 1) + \
                (~pos & (y == 0)).sum() / max((y == 0).sum(), 1)
        return float(max(grid, key=j))

    g = f[ok]
    res = {"n_images_task2": len(imgs), "n_with_task1_mask_and_task2_gt": len(ids) + len(overlap),
           "n_overlap_with_ham_excluded": len(overlap),
           "n_used": int(ok.sum()), "n_extract_errors": int((~ok).sum())}
    net_t = youden(g.d_net_score, g.gt_pigment_network, np.round(np.arange(0.0, 0.6, 0.01), 3))
    str_t = youden(g.d_streak_score, g.gt_streaks, np.round(np.arange(0.0, 0.6, 0.01), 3))
    blob = g.d_n_dots + g.d_n_globules
    glob_t = youden(g.d_n_globules, g.gt_globules, list(range(1, 30)), ge=True)
    dots_t = youden(g.d_n_dots, g.gt_globules, list(range(1, 30)), ge=True)
    for name, score, y in [("pigment_network", g.d_net_score, g.gt_pigment_network),
                           ("streaks", g.d_streak_score, g.gt_streaks),
                           ("globules(dots+globules blobs)", blob, g.gt_globules),
                           ("globules(large blobs)", g.d_n_globules, g.gt_globules)]:
        res[f"auc_{name}"] = round(float(roc_auc_score(y, score)), 3)
        res[f"prevalence_{name}"] = round(float(y.mean()), 3)
    res["thresholds"] = {"d_network": net_t, "d_streaks": str_t, "d_globules": int(glob_t),
                         "d_dots": int(dots_t)}
    (OUT / "task2_d_calibration.json").write_text(json.dumps(res, indent=2))
    t = replace(load_thresholds(), d_network=net_t, d_streaks=str_t,
                d_globules=max(1, int(glob_t)), d_dots=max(1, int(dots_t)))
    prov = json.loads(THRESH_FILE.read_text())["provenance"] if THRESH_FILE.exists() else {}
    prov["D"] = "ISIC 2018 Task 2 attribute masks (HAM overlap removed), Youden's J; task2_d_calibration.json"
    save_thresholds(t, prov)
    print(json.dumps(res, indent=1))


# --------------------------------------------------------------------- HAM10000

def step_ham() -> None:
    ham = DATA_DIR / "ham10000"
    meta = pd.read_csv(ham / "HAM10000_metadata.tab", sep=None, engine="python")
    zm = zipfile.ZipFile(ham / "HAM10000_segmentations_lesion_tschandl.zip")
    masks = {n.rsplit("/", 1)[-1].replace("_segmentation.png", ""): n
             for n in zm.namelist() if n.endswith(".png")}
    zips = [zipfile.ZipFile(ham / f"HAM10000_images_part_{i}.zip") for i in (1, 2)]
    imgs = {}
    for z in zips:
        for n in z.namelist():
            if n.endswith(".jpg"):
                imgs[n.rsplit("/", 1)[-1][:-4]] = (z, n)
    t = load_thresholds()

    def jobs():
        for iid in meta.image_id:
            z, n = imgs[iid]
            rgb = np.array(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
            m = np.array(Image.open(io.BytesIO(zm.read(masks[iid]))).convert("L")) > 127
            yield (iid, rgb, m, t)

    with ProcessPoolExecutor(8) as ex:
        feats = pd.DataFrame(list(ex.map(_safe_extract, jobs(), chunksize=16)))
    f = meta.merge(feats.rename(columns={"key": "image_id"}), on="image_id",
                   how="left", validate="one_to_one")
    f.to_parquet(PREP / "ham10000_features.parquet")
    summ = {"n_meta": int(len(meta)), "n_images_found": int(sum(i in imgs for i in meta.image_id)),
            "n_masks_found": int(sum(i in masks for i in meta.image_id)),
            "n_extract_errors": int(f.error.notna().sum()),
            "errors_sample": f.loc[f.error.notna(), ["image_id", "error"]].head(5).values.tolist(),
            "thresholds": asdict(t)}
    (OUT / "ham10000_extraction_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=1, default=str))


# ----------------------------------------------------------- PH2 D validation

def step_ph2val() -> None:
    f = pd.read_parquet(PREP / "ph2_features_raw.parquet")
    t = load_thresholds()
    z = zipfile.ZipFile(DATA_DIR / "ph2" / "PH2.zip")
    feats = _run([(n, *_ph2_pair(z, n), t) for n in f.name])
    g = f[["name", "clinical_dx", "asymmetry", "colors", "dots_globules", "streaks",
           "regression", "blue_white_veil", "pigment_network", "expert_C", "A_oof", "C_oof", "fold"]] \
        .merge(feats.rename(columns={"key": "name"}), on="name", validate="one_to_one")
    cnn = PREP / "ph2_structure_cnn.parquet"
    if cnn.exists():
        g = rescore(g.merge(pd.read_parquet(cnn), on="name", validate="one_to_one"), t)
    y_dg = (g.dots_globules != "A").astype(int)
    y_st = (g.streaks == "P").astype(int)
    pred_dg = ((g.d_has_dots == 1) | (g.d_has_globules == 1)).astype(int)
    # PH2 external TDS uses OOF A and C (never tuned on the image itself)
    g["TDS_oof"] = 1.3 * g.A_oof + 0.1 * g.B + 0.5 * g.C_oof + 0.5 * g.D
    res = {
        "dots_globules": {"prevalence": int(y_dg.sum()),
                          "kappa": round(float(cohen_kappa_score(y_dg, pred_dg)), 3),
                          "auc_blob_count": round(float(roc_auc_score(y_dg, g.d_n_dots + g.d_n_globules)), 3)},
        "streaks": {"prevalence": int(y_st.sum()),
                    "kappa": round(float(cohen_kappa_score(y_st, g.d_has_streaks)), 3),
                    "auc_streak_score": round(float(roc_auc_score(y_st, g.d_streak_score)), 3)},
        "D_distribution": g.D.value_counts().sort_index().to_dict(),
        "B_distribution": g.B.value_counts().sort_index().to_dict(),
        "TDS_oof_by_dx": g.groupby("clinical_dx").TDS_oof.describe().round(2).to_dict("index"),
    }
    g.to_parquet(PREP / "ph2_features.parquet")
    (OUT / "ph2_d_validation.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


# ------------------------------------------------------------------ finalise

def rescore(f: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    """Recompute A, C, D, TDS from stored continuous features (B is unchanged).
    A ignores the (minor) effect of thresholds on the choice of axis pair."""
    f = f.copy()
    f["A"] = a_score(f, t.a_shape, t.a_color)
    f["C"] = c_score(f, t.c_min_frac_per or t.c_min_frac)
    has = {"network": f.d_net_score > t.d_network,
           "structureless": f.d_structureless_frac > t.d_structureless,
           "streaks": f.d_streak_score > t.d_streaks,
           "dots": f.d_n_dots >= t.d_dots,
           "globules": f.d_n_globules >= t.d_globules}
    for k, v in has.items():
        f[f"d_has_{k}"] = v.astype(int)
    f["D_heuristic"] = np.maximum(1, sum(v.astype(int) for v in has.values()))
    if "cnn_has_pigment_network" in f:
        # learned detectors (structure_cnn.py) replace the weak heuristics; Task 2
        # labels dots and globules jointly, so they count as one structure (D range 1-4)
        f["D"] = np.maximum(1, f.cnn_has_pigment_network + f.cnn_has_streaks
                            + f.cnn_has_globules + f.d_has_structureless)
    else:
        f["D"] = f.D_heuristic
    f["TDS"] = (1.3 * f.A + 0.1 * f.B + 0.5 * f.C + 0.5 * f.D).round(2)
    return f


DERM_FEATURES = {
    "A": ("A", "Stolz asymmetry 0-2: axes (of the least-asymmetric orthogonal pair) where contour XOR > a_shape or half-colour ΔE > a_color"),
    "B": ("B", "Stolz border 0-8: 45° sectors whose pigment ramp is narrower than b_ramp_frac × equivalent diameter (heuristic)"),
    "C": ("C", "Stolz colours 1-6: Stolz colours whose pixel fraction ≥ per-colour PH2-calibrated threshold"),
    "D": ("D", "Stolz structures 1-4: network, streaks, dots/globules (learned CNN detectors, joint) + structureless (heuristic)"),
    "D_heuristic": ("D", "Comparison only: D from the hand-crafted detectors (weak on Task 2)"),
    "cnn_p_pigment_network": ("D", "CNN probability that pigment network is present (ResNet18, ISIC 2018 Task 2)"),
    "cnn_p_globules": ("D", "CNN probability that dots/globules are present"),
    "cnn_p_streaks": ("D", "CNN probability that streaks are present"),
    "cnn_p_negative_network": ("D*", "CNN probability of negative network (not a Stolz structure)"),
    "cnn_p_milia_like_cyst": ("D*", "CNN probability of milia-like cysts (not a Stolz structure)"),
    "TDS": ("rule", "1.3A + 0.1B + 0.5C + 0.5D"),
    "a_shape_max": ("A", "Max contour asymmetry (XOR area / area) over the two axes"),
    "a_color_max": ("A", "Max ΔE76 between mean Lab of the two halves"),
    "b_ramp_mean": ("B", "Mean pigment-ramp width / equivalent diameter over 8 sectors (low = abrupt)"),
    "b_ramp_std": ("B", "Std of sector ramp widths (irregularity of border sharpness)"),
    "b_compactness": ("B", "perimeter² / (4π area); 1 = circle"),
    "c_L_std": ("C", "Std of L* inside lesion (colour variegation)"),
    "c_ab_std": ("C", "Mean std of a*, b* inside lesion"),
    "d_net_score": ("D", "Fraction of lesion covered by dark mesh lines (black-hat)"),
    "d_n_dots": ("D", "Count of small dark blobs (LoG σ<2.5 px)"),
    "d_n_globules": ("D", "Count of larger dark blobs (LoG σ≥2.5 px)"),
    "d_streak_score": ("D", "Fraction of strong peripheral ridges oriented radially"),
    "d_structureless_frac": ("D", "Fraction of lesion with low local texture and no mesh"),
    "lesion_area_frac": ("size", "Lesion area / image area (no mm calibration in HAM10000)"),
    "age": ("context", "Patient age"), "sex": ("context", "Patient sex"),
    "localization": ("context", "Anatomic site"),
}
DERM_EXCLUDED = {
    "dx": "TARGET source", "dx_type": "LEAK: histo means the lesion was excised (suspicious)",
    "dataset": "SHORTCUT: source collection/device (MoleMax ≈ follow-up nevi); robustness strata only",
    "lesion_id": "grouping key only", "image_id": "identifier",
}


def step_finalize() -> None:
    t = load_thresholds()
    f = pd.read_parquet(PREP / "ham10000_features.parquet")
    counts = [("HAM10000 images", len(f), int((f.dx == "mel").sum()))]
    # demo scope: structure CNN was run on the primary cohort, one image per lesion
    f = f.merge(pd.read_parquet(PREP / "ham10000_structure_cnn.parquet"), on="image_id",
                how="inner", validate="one_to_one")
    counts.append(("primary cohort, one image per lesion (histo mel+nv; CNN applied)",
                   len(f), int((f.dx == "mel").sum())))
    f = rescore(f, t)
    f = f[f.error.isna()].copy()
    counts.append(("feature extraction succeeded", len(f), int((f.dx == "mel").sum())))
    f["y_melanoma"] = np.where(f.dx == "mel", 1.0, np.where(f.dx == "nv", 0.0, np.nan))
    f["y_mel_vs_benign"] = np.where(f.dx == "mel", 1.0,
                                    np.where(f.dx.isin(["nv", "bkl", "df", "vasc"]), 0.0, np.nan))
    f["histo"] = (f.dx_type == "histo").astype(int)
    f["primary_cohort"] = ((f.histo == 1) & f.dx.isin(["mel", "nv"])).astype(int)
    p = f[f.primary_cohort == 1]
    counts.append(("primary cohort: histology-confirmed melanocytic (mel+nv)", len(p), int((p.dx == "mel").sum())))
    counts.append(("  unique lesions in primary cohort", int(p.lesion_id.nunique()),
                   int(p.loc[p.dx == "mel", "lesion_id"].nunique())))
    s = f[f.y_mel_vs_benign.notna()]
    counts.append(("secondary: melanoma vs all benign (any confirmation)", len(s), int((s.dx == "mel").sum())))
    f["age_missing"] = f.age.isna().astype(int)
    f.drop(columns=["error"]).to_parquet(PREP / "ham10000_model.parquet", index=False)
    pd.DataFrame(counts, columns=["step", "rows", "melanoma"]).to_csv(OUT / "ham10000_row_counts.csv", index=False)

    rows = []
    for c, (crit, desc) in DERM_FEATURES.items():
        rows.append({"feature": c, "criterion": crit, "definition": desc,
                     "source": "HAM10000 image + Tschandl mask (src/skin_cancer/prep/abcd_image.py)"
                     if crit not in ("context",) else "HAM10000 metadata",
                     "type": str(f[c].dtype), "missing_frac": round(float(f[c].isna().mean()), 4),
                     "missing_treatment": "indicator + median impute inside CV folds" if c == "age"
                     else ("NaN when sector has no border pixels; median impute in folds" if c.startswith("b_ramp") else "none needed"),
                     "decision_time": "Y: computed from the dermoscopic image at the visit" if crit != "context"
                     else "Y: visible/askable at the visit"})
    for n in COLOR_NAMES:
        rows.append({"feature": f"c_frac_{n}", "criterion": "C", "definition": f"Fraction of lesion pixels nearest the {n} reference (skin-normalised Lab)",
                     "source": "image", "type": "float64", "missing_frac": 0.0, "missing_treatment": "none needed",
                     "decision_time": "Y: computed from the dermoscopic image at the visit"})
    pd.DataFrame(rows).to_csv(OUT / "dermoscopy_data_dictionary.csv", index=False)
    (OUT / "dermoscopy_excluded.json").write_text(json.dumps(DERM_EXCLUDED, indent=2))
    dist = {"primary_cohort_by_dx": {k: {c: g[c].value_counts().sort_index().to_dict()
                                         for c in ["A", "B", "C", "D"]}
                                     for k, g in p.groupby("dx")},
            "primary_TDS_by_dx": p.groupby("dx").TDS.describe().round(2).to_dict("index")}
    (OUT / "ham10000_score_distributions.json").write_text(json.dumps(dist, indent=2, default=str))
    print(pd.DataFrame(counts).to_string(), json.dumps(dist["primary_TDS_by_dx"], indent=1))


STEPS = {"ph2": step_ph2, "task2": step_task2, "ham": step_ham, "ph2val": step_ph2val,
         "finalize": step_finalize}

if __name__ == "__main__":
    for s in sys.argv[1:] or list(STEPS):
        STEPS[s]()
