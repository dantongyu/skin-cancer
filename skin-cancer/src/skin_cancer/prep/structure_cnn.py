"""Learned detectors for Stolz D structures (concept bottleneck).

A ResNet18 (ImageNet-pretrained) is fine-tuned on lesion crops from ISIC 2018 Task 2
to predict image-level presence of 5 expert-annotated dermoscopic attributes. Only
"is structure X present?" is learned; the TDS arithmetic stays transparent. Grad-CAM
on the same network gives the demo's "where" evidence.

Steps:
  cache   crop lesions (mask bbox + margin) to 384x384 for Task 2 -> uint8 npy
  train   80/20 stratified split, BCE with pos_weight, pick per-structure thresholds
          on the validation 20% (Youden), report validation AUCs
  infer   apply to HAM10000 and PH2, write per-image structure probabilities
  ph2val  external validation on PH2 structure labels

Usage: uv run python -m skin_cancer.prep.structure_cnn [cache train infer ph2val]
"""
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split

from skin_cancer.config import DATA_DIR, REPORTS
from skin_cancer.prep.abcd_image import clean_mask

OUT = REPORTS / "stage3"
PREP = DATA_DIR / "prepared"
MODELS = DATA_DIR / "models"
ATTRS = ["pigment_network", "globules", "streaks", "negative_network", "milia_like_cyst"]
SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def lesion_crop(rgb: np.ndarray, mask: np.ndarray, size: int = SIZE, margin: float = 0.15):
    """Square crop around the lesion bounding box, resized to size x size."""
    s = 1024 / max(rgb.shape[:2])        # downscale first: full-res mask cleaning is slow
    if s < 1:
        dsize = (round(rgb.shape[1] * s), round(rgb.shape[0] * s))
        rgb = cv2.resize(rgb, dsize, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST) > 0
    m = clean_mask(mask)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        ys, xs = np.array([0, rgb.shape[0] - 1]), np.array([0, rgb.shape[1] - 1])
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + margin)
    half = max(half, 16)
    y0, y1 = int(max(0, cy - half)), int(min(rgb.shape[0], cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(rgb.shape[1], cx + half))
    return cv2.resize(rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def build_model() -> nn.Module:
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(ATTRS))
    return m


def to_tensor(batch_uint8: np.ndarray) -> torch.Tensor:
    x = (batch_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------- cache

_Z: dict = {}


def _cache_job(args):
    t2, img_name, mask_name, gt_names = args
    if not _Z:
        _Z["i"] = zipfile.ZipFile(f"{t2}/ISIC2018_Task1-2_Training_Input.zip")
        _Z["m"] = zipfile.ZipFile(f"{t2}/ISIC2018_Task1_Training_GroundTruth.zip")
        _Z["g"] = zipfile.ZipFile(f"{t2}/ISIC2018_Task2_Training_GroundTruth_v3.zip")
    img = Image.open(io.BytesIO(_Z["i"].read(img_name)))
    img.draft("RGB", (1024, 1024))       # fast JPEG decode at reduced scale
    rgb = np.array(img.convert("RGB"))
    m = np.array(Image.open(io.BytesIO(_Z["m"].read(mask_name))).convert("L")) > 127
    m = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    y = [np.asarray(Image.open(io.BytesIO(_Z["g"].read(n)))).max() > 0 for n in gt_names]
    return lesion_crop(rgb, m), np.array(y)


def step_cache() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    t2 = DATA_DIR / "isic2018_task2"
    zi = zipfile.ZipFile(t2 / "ISIC2018_Task1-2_Training_Input.zip")
    zm = zipfile.ZipFile(t2 / "ISIC2018_Task1_Training_GroundTruth.zip")
    zg = zipfile.ZipFile(t2 / "ISIC2018_Task2_Training_GroundTruth_v3.zip")
    imgs = {n.rsplit("/", 1)[-1][:-4]: n for n in zi.namelist() if n.endswith(".jpg")}
    masks = {n.rsplit("/", 1)[-1].replace("_segmentation.png", ""): n
             for n in zm.namelist() if n.endswith(".png")}
    gt = {}
    for n in zg.namelist():
        if n.endswith(".png"):
            iid, attr = n.rsplit("/", 1)[-1][:-4].split("_attribute_")
            gt.setdefault(iid, {})[attr] = n
    ids = sorted(set(imgs) & set(masks) & set(gt))
    jobs = [(str(t2), imgs[i], masks[i], [gt[i][a] for a in ATTRS]) for i in ids]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_cache_job, jobs, chunksize=8))
    X = np.stack([r[0] for r in res])
    Y = np.stack([r[1] for r in res]).astype(np.uint8)
    np.save(PREP / "task2_crops.npy", X)
    np.save(PREP / "task2_labels.npy", Y)
    pd.Series(ids).to_csv(PREP / "task2_ids.csv", index=False, header=["isic_id"])
    print("cached", X.shape, "label prevalence", dict(zip(ATTRS, Y.mean(0).round(3))))


# ----------------------------------------------------------------------- train

def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    for i in range(len(x)):
        k = rng.integers(4)
        img = np.rot90(x[i], k)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        img = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
        x[i] = np.clip(img, 0, 255).astype(np.uint8)
    return x


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, bs: int = 64) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        xb = to_tensor(X[i:i + bs]).to(dev)
        # test-time augmentation: average over the 4 rotations
        p = sum(torch.sigmoid(model(torch.rot90(xb, k, dims=(2, 3)))) for k in range(4)) / 4
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def step_train(epochs: int = 8, bs: int = 32, seed: int = 0) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.load(PREP / "task2_crops.npy")
    Y = np.load(PREP / "task2_labels.npy").astype(np.float32)
    strat = Y[:, ATTRS.index("streaks")] * 2 + Y[:, ATTRS.index("globules")]
    tr, va = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=strat)
    dev = device()
    model = build_model().to(dev)
    pos = Y[tr].mean(0)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(tr) // bs + 1))
    log, best, best_state = [], -1, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = to_tensor(augment(X[idx], rng)).to(dev)
            yb = torch.from_numpy(Y[idx]).to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        pv = predict(model, X[va])
        aucs = {a: float(roc_auc_score(Y[va, j], pv[:, j])) for j, a in enumerate(ATTRS)}
        mean_auc = float(np.mean([aucs[a] for a in ["pigment_network", "globules", "streaks"]]))
        log.append({"epoch": ep, "train_loss": tot / len(tr), **{f"val_auc_{a}": round(v, 4) for a, v in aucs.items()}})
        print(log[-1], flush=True)
        if mean_auc > best:
            best, best_state = mean_auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS / "structure_resnet18.pt")
    pv = predict(model.to(dev), X[va])

    thr, val = {}, {}
    for j, a in enumerate(ATTRS):
        y, p = Y[va, j], pv[:, j]
        grid = np.linspace(0.02, 0.98, 97)
        J = [((p >= g) & (y == 1)).sum() / max(y.sum(), 1) + ((p < g) & (y == 0)).sum() / max((1 - y).sum(), 1) - 1 for g in grid]
        thr[a] = float(grid[int(np.argmax(J))])
        val[a] = {"val_auc": round(float(roc_auc_score(y, p)), 3), "val_prevalence": int(y.sum()),
                  "threshold": round(thr[a], 3), "youden_J": round(float(max(J)), 3),
                  "sens": round(float(((p >= thr[a]) & (y == 1)).sum() / max(y.sum(), 1)), 3),
                  "spec": round(float(((p < thr[a]) & (y == 0)).sum() / max((1 - y).sum(), 1)), 3)}
    res = {"n_train": len(tr), "n_val": len(va), "epochs": epochs, "best_epoch_by_mean_val_auc":
           int(np.argmax([np.mean([r[f"val_auc_{a}"] for a in ["pigment_network", "globules", "streaks"]]) for r in log])),
           "device": str(dev), "validation": val, "training_log": log}
    (OUT / "structure_cnn_training.json").write_text(json.dumps(res, indent=2))
    (MODELS / "structure_thresholds.json").write_text(json.dumps(thr, indent=2))
    print(json.dumps(val, indent=1))


# ----------------------------------------------------------------------- infer

def load_trained() -> tuple[nn.Module, dict]:
    m = build_model()
    m.load_state_dict(torch.load(MODELS / "structure_resnet18.pt", map_location="cpu"))
    return m.to(device()), json.loads((MODELS / "structure_thresholds.json").read_text())


def step_infer() -> None:
    model, thr = load_trained()
    ham = DATA_DIR / "ham10000"
    meta = pd.read_csv(ham / "HAM10000_metadata.tab", sep=None, engine="python")
    zm = zipfile.ZipFile(ham / "HAM10000_segmentations_lesion_tschandl.zip")
    masks = {n.rsplit("/", 1)[-1].replace("_segmentation.png", ""): n
             for n in zm.namelist() if n.endswith(".png")}
    imgs = {}
    for i in (1, 2):
        z = zipfile.ZipFile(ham / f"HAM10000_images_part_{i}.zip")
        for n in z.namelist():
            if n.endswith(".jpg"):
                imgs[n.rsplit("/", 1)[-1][:-4]] = (z, n)

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(8)           # cv2/PIL release the GIL

    def crops(ids, getter):
        for i in range(0, len(ids), 256):
            chunk = ids[i:i + 256]
            yield chunk, np.stack(list(pool.map(lambda k: lesion_crop(*getter(k)), chunk)))

    def ham_get(iid):
        z, n = imgs[iid]
        rgb = np.array(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
        return rgb, np.array(Image.open(io.BytesIO(zm.read(masks[iid]))).convert("L")) > 127

    zp = zipfile.ZipFile(DATA_DIR / "ph2" / "PH2.zip")
    ph2_ids = sorted(n.rsplit("/", 1)[-1][:-4] for n in zp.namelist()
                     if n.startswith("ph2_dataset/trainx/") and n.endswith(".bmp"))

    def ph2_get(name):
        rgb = np.array(Image.open(io.BytesIO(zp.read(f"ph2_dataset/trainx/{name}.bmp"))).convert("RGB"))
        return rgb, np.array(Image.open(io.BytesIO(zp.read(f"ph2_dataset/trainy/{name}_lesion.bmp")))
                             .convert("L")) > 127

    # demo scope: primary cohort only (histology-confirmed mel + nv), one image per lesion
    prim = meta[(meta.dx_type == "histo") & meta.dx.isin(["mel", "nv"])] \
        .sort_values("image_id").drop_duplicates("lesion_id")
    for name, ids, getter, key in [("ham10000", list(prim.image_id), ham_get, "image_id"),
                                   ("ph2", ph2_ids, ph2_get, "name")]:
        rows = []
        for chunk, X in crops(ids, getter):
            p = predict(model, X)
            for k, pr in zip(chunk, p):
                rows.append({key: k, **{f"cnn_p_{a}": float(v) for a, v in zip(ATTRS, pr)},
                             **{f"cnn_has_{a}": int(v >= thr[a]) for a, v in zip(ATTRS, pr)}})
            print(name, len(rows), flush=True)
        pd.DataFrame(rows).to_parquet(PREP / f"{name}_structure_cnn.parquet", index=False)


# ---------------------------------------------------------------------- ph2val

def step_ph2val() -> None:
    from skin_cancer.data.profile_dermoscopy import load_ph2
    df = load_ph2().merge(pd.read_parquet(PREP / "ph2_structure_cnn.parquet"), on="name",
                          validate="one_to_one")
    y_dg = (df.dots_globules != "A").astype(int)
    y_st = (df.streaks == "P").astype(int)
    y_net_atyp = (df.pigment_network == "AT").astype(int)
    res = {
        "dots_globules": {"prevalence": int(y_dg.sum()), "n": len(df),
                          "auc": round(float(roc_auc_score(y_dg, df.cnn_p_globules)), 3),
                          "kappa_at_threshold": round(float(cohen_kappa_score(y_dg, df.cnn_has_globules)), 3)},
        "streaks": {"prevalence": int(y_st.sum()),
                    "auc": round(float(roc_auc_score(y_st, df.cnn_p_streaks)), 3),
                    "kappa_at_threshold": round(float(cohen_kappa_score(y_st, df.cnn_has_streaks)), 3)},
        "pigment_network": {"note": "PH2 marks network present (typical/atypical) in all 200; only detection rate is checkable",
                            "detected_frac": round(float(df.cnn_has_pigment_network.mean()), 3),
                            "auc_atypical_vs_typical": round(float(roc_auc_score(y_net_atyp, df.cnn_p_pigment_network)), 3)},
    }
    (OUT / "structure_cnn_ph2_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


STEPS = {"cache": step_cache, "train": step_train, "infer": step_infer, "ph2val": step_ph2val}

if __name__ == "__main__":
    for s in sys.argv[1:] or list(STEPS):
        STEPS[s]()


def gradcam(model: nn.Module, crop: np.ndarray, attr: str) -> np.ndarray:
    """Grad-CAM heat-map (0-1, crop resolution) for one structure on layer4."""
    model.eval()
    dev = next(model.parameters()).device
    store = {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        with torch.enable_grad():
            x = to_tensor(crop[None]).to(dev)
            out = model(x)
            model.zero_grad()
            out[0, ATTRS.index(attr)].backward()
        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * store["a"]).sum(1))[0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    cam = cam / (cam.max() + 1e-8)
    return cv2.resize(cam, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
