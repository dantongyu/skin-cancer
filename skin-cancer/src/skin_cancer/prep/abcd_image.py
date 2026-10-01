"""Dermoscopic ABCD (Stolz) feature extraction from an RGB image + lesion mask.

Every criterion returns both a score and the evidence used to reach it, so the
web demo can draw it (axes, abrupt border sectors, colour map, detected
structures). Thresholds live in `Thresholds`; the A/C ones are calibrated on PH2
expert labels and the D ones on ISIC 2018 Task 2 masks (see prep/dermoscopy.py).
B has no public ground truth and uses a documented heuristic.

Stolz TDS = 1.3*A + 0.1*B + 0.5*C + 0.5*D
    A 0-2  axes of asymmetry (contour, colour)
    B 0-8  border sectors with abrupt pigment cut-off
    C 1-6  colours among white, red, light brown, dark brown, blue-gray, black
    D 1-5  structures among network, structureless area, streaks, dots, globules
"""
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color as skcolor
from skimage.feature import blob_log
from skimage.filters import sato

WORK_SIZE = 512  # longest side after resizing; all pixel constants assume this

COLOR_NAMES = ["white", "red", "light_brown", "dark_brown", "blue_gray", "black"]
# Images are normalised so the perilesional skin has this CIELAB colour (median
# skin of PH2, reports/stage3/ph2_lab_reference.txt). Colour references below are
# in that skin-referenced space; light/dark brown are anchored on PH2 lesions
# whose experts listed only that brown. "skin" is a pseudo-class that absorbs
# non-pigmented pixels inside the mask and is never counted as a colour.
CANON_SKIN_LAB = np.array([74.0, 14.0, 18.0])
COLOR_CENTROIDS_LAB = np.array([
    [84.0, 6.0, 8.0],     # white   (lighter than skin, low chroma)
    [52.0, 38.0, 22.0],   # red     (high a*)
    [61.0, 23.0, 36.0],   # light brown (PH2 'light-brown only' lesions)
    [47.0, 19.0, 24.0],   # dark brown  (PH2 'dark-brown only', darker half)
    [50.0, 2.0, 0.0],     # blue-gray (chroma near zero / bluish)
    [22.0, 4.0, 4.0],     # black
])
SKIN_CENTROID_LAB = CANON_SKIN_LAB


@dataclass
class Thresholds:
    # A: an axis is asymmetric if contour OR colour asymmetry exceeds these
    a_shape: float = 0.20      # XOR area / lesion area for the flip about the axis
    a_color: float = 8.0       # ΔE76 between the mean Lab colour of the two halves
    # B: a sector is "abrupt" if the pigment ramp is narrower than this
    # fraction of the lesion's equivalent diameter (heuristic, no ground truth)
    b_ramp_frac: float = 0.03
    # C: a colour is present if it covers at least this fraction of the lesion
    c_min_frac: float = 0.05
    # optional per-colour presence thresholds (overrides c_min_frac), same order as COLOR_NAMES
    c_min_frac_per: list | None = None
    c_l_offset: float = 0.0    # global L* shift applied to the centroids
    # D: structure presence cut-offs
    d_network: float = 0.10    # fraction of lesion covered by dark mesh lines
    d_dots: int = 6            # small dark blobs
    d_globules: int = 3        # larger dark blobs
    d_streaks: float = 0.12    # fraction of peripheral ridge pixels oriented radially
    d_structureless: float = 0.30  # fraction of lesion with low texture


@dataclass
class ABCDResult:
    A: int
    B: int
    C: int
    D: int
    TDS: float
    features: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def flat(self) -> dict:
        return {"A": self.A, "B": self.B, "C": self.C, "D": self.D, "TDS": self.TDS,
                **self.features}


# --------------------------------------------------------------------------- utils

def resize_pair(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    s = WORK_SIZE / max(h, w)
    size = (round(w * s), round(h * s))
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return rgb, mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return ndi.binary_fill_holes(lab == (np.argmax(sizes) + 1))


def remove_hair(rgb: np.ndarray) -> np.ndarray:
    """DullRazor-style: black-hat finds thin dark hairs, inpaint over them."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, hair = cv2.threshold(bh, 12, 255, cv2.THRESH_BINARY)
    hair = cv2.dilate(hair, np.ones((3, 3), np.uint8))
    return cv2.inpaint(rgb, hair, 5, cv2.INPAINT_TELEA)


def shades_of_gray(rgb: np.ndarray, p: int = 6) -> np.ndarray:
    x = rgb.astype(np.float64)
    illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1 / p)
    illum = illum / np.linalg.norm(illum) * np.sqrt(3)
    return np.clip(x / illum, 0, 255).astype(np.uint8)


def skin_normalise(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Translate Lab so the perilesional skin ring matches CANON_SKIN_LAB."""
    d = ndi.distance_transform_edt(~mask)
    ring = (d > 5) & (d < 25)
    if ring.sum() < 50:
        ring = ~mask
    return lab - np.median(lab[ring], axis=0) + CANON_SKIN_LAB


def preprocess(rgb: np.ndarray, mask: np.ndarray):
    rgb, mask = resize_pair(rgb, mask)
    mask = clean_mask(mask)
    rgb = remove_hair(rgb)
    lab = skin_normalise(skcolor.rgb2lab(rgb), mask)
    return rgb, mask, lab


def rotate(img: np.ndarray, center, angle_deg: float, nearest: bool) -> np.ndarray:
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=flag,
                          borderMode=cv2.BORDER_CONSTANT)


def pad_center(mask: np.ndarray, lab: np.ndarray):
    """Pad so the lesion centroid sits at the canvas centre (lossless rotation)."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    r = int(np.ceil(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()))) + 2
    size = 2 * r + 1
    m = np.zeros((size, size), np.uint8)
    L = np.zeros((size, size, 3), np.float32)
    y0, x0 = int(round(cy)) - r, int(round(cx)) - r
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(mask.shape[0], y0 + size), min(mask.shape[1], x0 + size)
    m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1]
    L[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = lab[sy0:sy1, sx0:sx1]
    return m, L, (cx, cy)


# ------------------------------------------------------------------- A asymmetry

def asymmetry(mask: np.ndarray, lab: np.ndarray, t: Thresholds):
    m, L, (cx, cy) = pad_center(mask, lab)
    c = (m.shape[1] / 2 - 0.5, m.shape[0] / 2 - 0.5)
    area = m.sum()
    best = None
    for ang in range(0, 180, 10):
        mr = rotate(m, c, ang, nearest=True).astype(bool)
        Lr = np.dstack([rotate(L[..., i], c, ang, nearest=False) for i in range(3)])
        axes = []
        for flip_axis in (0, 1):  # 0: flip top/bottom (horizontal axis), 1: left/right
            shape = np.logical_xor(mr, np.flip(mr, axis=flip_axis)).sum() / area
            mid = mr.shape[flip_axis] // 2
            sl1 = [slice(None)] * 2
            sl2 = [slice(None)] * 2
            sl1[flip_axis], sl2[flip_axis] = slice(0, mid), slice(mid, None)
            h1, h2 = mr[tuple(sl1)], mr[tuple(sl2)]
            if h1.sum() < 10 or h2.sum() < 10:
                col = 0.0
            else:
                col = float(np.linalg.norm(Lr[tuple(sl1)][h1].mean(0) - Lr[tuple(sl2)][h2].mean(0)))
            axes.append((shape, col))
        score = sum(s / t.a_shape + c_ / t.a_color for s, c_ in axes)
        if best is None or score < best[0]:
            best = (score, ang, axes)
    _, ang, axes = best
    flags = [s > t.a_shape or c_ > t.a_color for s, c_ in axes]
    feats = {
        "a_shape_ax1": axes[0][0], "a_shape_ax2": axes[1][0],
        "a_color_ax1": axes[0][1], "a_color_ax2": axes[1][1],
        "a_shape_max": max(a[0] for a in axes), "a_color_max": max(a[1] for a in axes),
    }
    ev = {"center_xy": [float(cx), float(cy)], "axis_angle_deg": float(ang),
          "axis_asymmetric": flags}
    return int(sum(flags)), feats, ev


# ----------------------------------------------------------------------- B border

def border(mask: np.ndarray, lab: np.ndarray, axis_angle: float, t: Thresholds):
    Ls = cv2.GaussianBlur(lab[..., 0].astype(np.float32), (0, 0), 1.5)
    gy, gx = np.gradient(Ls)
    grad = np.hypot(gx, gy)
    m8 = mask.astype(np.uint8)
    edge = (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))).astype(bool)
    eq_diam = 2 * np.sqrt(mask.sum() / np.pi)
    band = max(3, int(0.06 * eq_diam))
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    ring_in = (d_in > 0) & (d_in <= band)
    ring_out = (d_out > 0) & (d_out <= band)
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.indices(mask.shape)
    theta = (np.degrees(np.arctan2(yy - cy, xx - cx)) - axis_angle) % 360
    sector = (theta // 45).astype(int)
    ramps, abrupt = [], []
    for s in range(8):
        sec = sector == s
        e = edge & sec
        if e.sum() < 3 or (ring_in & sec).sum() < 3 or (ring_out & sec).sum() < 3:
            ramps.append(np.nan)
            abrupt.append(False)
            continue
        contrast = abs(np.median(Ls[ring_out & sec]) - np.median(Ls[ring_in & sec]))
        g = np.mean(grad[e]) + 1e-6
        ramp = contrast / g  # pixels needed to climb the in/out step at edge slope
        ramps.append(ramp / eq_diam)
        abrupt.append(bool(ramp / eq_diam < t.b_ramp_frac and contrast > 3))
    ramps_arr = np.array(ramps, dtype=float)
    perim = cv2.arcLength(max(cv2.findContours(m8, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)[0],
                              key=cv2.contourArea), True)
    feats = {
        "b_ramp_mean": float(np.nanmean(ramps_arr)) if np.isfinite(ramps_arr).any() else np.nan,
        "b_ramp_min": float(np.nanmin(ramps_arr)) if np.isfinite(ramps_arr).any() else np.nan,
        "b_ramp_std": float(np.nanstd(ramps_arr)) if np.isfinite(ramps_arr).any() else np.nan,
        "b_compactness": float(perim ** 2 / (4 * np.pi * mask.sum())),
    }
    ev = {"sector_abrupt": abrupt, "sector_ramp_frac": [None if np.isnan(r) else float(r)
                                                        for r in ramps]}
    return int(sum(abrupt)), feats, ev


# ----------------------------------------------------------------------- C colour

def colours(mask: np.ndarray, lab: np.ndarray, t: Thresholds):
    px = lab[mask]
    cents = np.vstack([COLOR_CENTROIDS_LAB + np.array([t.c_l_offset, 0, 0]),
                       SKIN_CENTROID_LAB])
    d = np.linalg.norm(px[:, None, :] - cents[None, :, :], axis=2)
    lab_idx = d.argmin(1)                       # 0-5 Stolz colours, 6 = skin
    frac = np.bincount(lab_idx, minlength=7)[:6] / len(lab_idx)
    thr = np.array(t.c_min_frac_per) if t.c_min_frac_per else np.full(6, t.c_min_frac)
    present = frac >= thr
    label_map = np.full(mask.shape, -1, np.int8)
    label_map[mask] = np.where(lab_idx == 6, -1, lab_idx)
    feats = {f"c_frac_{n}": float(f) for n, f in zip(COLOR_NAMES, frac)}
    feats.update({"c_L_std": float(px[:, 0].std()), "c_ab_std": float(px[:, 1:].std(0).mean()),
                  "c_L_mean": float(px[:, 0].mean())})
    ev = {"colors_present": [n for n, p in zip(COLOR_NAMES, present) if p],
          "color_fracs": dict(zip(COLOR_NAMES, frac.round(4).tolist())),
          "label_map": label_map}
    return max(1, int(present.sum())), feats, ev


# ------------------------------------------------------------ D structures

def structures(mask: np.ndarray, lab: np.ndarray, t: Thresholds):
    L = lab[..., 0].astype(np.float32)
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    area = mask.sum()
    inner = ndi.binary_erosion(mask, iterations=3)

    # pigment network: dark mesh (black-hat) enclosing light holes
    bh = cv2.morphologyEx(L8, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mesh = (bh > 8) & inner
    net_score = float(mesh.sum() / max(inner.sum(), 1))  # dark-line mesh density

    # dots (small) and globules (larger): dark blobs via LoG on inverted L
    inv = (255 - L8).astype(np.float32) / 255.0
    blobs = blob_log(inv, min_sigma=1.0, max_sigma=7.0, num_sigma=7, threshold=0.06)
    blobs = [b for b in blobs if inner[int(b[0]), int(b[1])]]
    dots = [b for b in blobs if b[2] < 2.5]
    globs = [b for b in blobs if b[2] >= 2.5]

    # streaks: dark ridges in the periphery oriented radially
    dist = ndi.distance_transform_edt(mask)
    periph = mask & (dist <= 0.2 * dist.max())
    ridge = sato(L / 100.0, sigmas=[1, 2], black_ridges=True)
    strong = periph & (ridge > np.percentile(ridge[mask], 90))
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    Ls = cv2.GaussianBlur(L, (0, 0), 1.5)
    gy, gx = np.gradient(Ls)
    sy, sx = np.nonzero(strong)
    if len(sy) > 20:
        # a ridge's gradient is perpendicular to it; radial ridge => gradient tangential
        rad = np.stack([sy - cy, sx - cx], 1)
        rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-6
        g = np.stack([gy[sy, sx], gx[sy, sx]], 1)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-6
        cosang = np.abs((rad * g).sum(1))
        streak_score = float((cosang < np.cos(np.radians(70))).mean())
    else:
        streak_score = 0.0
    streak_pts = np.stack([sy, sx], 1) if len(sy) else np.zeros((0, 2))

    # structureless: low local texture, outside mesh
    lstd = np.sqrt(np.maximum(ndi.uniform_filter(L ** 2, 7) - ndi.uniform_filter(L, 7) ** 2, 0))
    smooth = inner & (lstd < 1.5) & ~mesh
    structless = float(smooth.sum() / area)

    present = {
        "network": net_score > t.d_network,
        "structureless": structless > t.d_structureless,
        "streaks": streak_score > t.d_streaks,
        "dots": len(dots) >= t.d_dots,
        "globules": len(globs) >= t.d_globules,
    }
    feats = {"d_net_score": net_score, "d_n_dots": len(dots), "d_n_globules": len(globs),
             "d_streak_score": streak_score, "d_structureless_frac": structless,
             **{f"d_has_{k}": int(v) for k, v in present.items()}}
    ev = {"structures_present": [k for k, v in present.items() if v],
          "dots_yx_r": [[float(b[0]), float(b[1]), float(b[2] * np.sqrt(2))] for b in dots],
          "globules_yx_r": [[float(b[0]), float(b[1]), float(b[2] * np.sqrt(2))] for b in globs],
          "network_mask": mesh, "structureless_mask": smooth, "streak_mask": strong}
    return max(1, int(sum(present.values()))), feats, ev


# ------------------------------------------------------------------------ entry

def extract(rgb: np.ndarray, mask: np.ndarray, t: Thresholds | None = None,
            keep_evidence: bool = False) -> ABCDResult:
    t = t or Thresholds()
    rgb, mask, lab = preprocess(rgb, mask)
    if mask.sum() < 200:
        raise ValueError("lesion mask too small")
    A, fa, ea = asymmetry(mask, lab, t)
    B, fb, eb = border(mask, lab, ea["axis_angle_deg"], t)
    C, fc, ec = colours(mask, lab, t)
    D, fd, ed = structures(mask, lab, t)
    tds = round(1.3 * A + 0.1 * B + 0.5 * C + 0.5 * D, 2)
    feats = {**fa, **fb, **fc, **fd,
             "lesion_area_frac": float(mask.sum() / mask.size)}
    ev = {}
    if keep_evidence:
        ev = {"A": ea, "B": eb, "C": ec, "D": ed, "rgb": rgb, "mask": mask}
    return ABCDResult(A, B, C, D, tds, feats, ev)


def thresholds_dict(t: Thresholds) -> dict:
    return asdict(t)
