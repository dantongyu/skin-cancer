"""ABCD Teaching Demo — local Streamlit app (educational prototype, not a medical device).

Run:  uv run streamlit run app/demo.py
"""
import io
import json
import zipfile

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage as ndi

from skin_cancer.config import DATA_DIR, REPORTS
from skin_cancer.data.profile_dermoscopy import load_ph2
from skin_cancer.model.train import DERM_CONT
from skin_cancer.prep.abcd_image import COLOR_NAMES, extract
from skin_cancer.prep.dermoscopy import load_thresholds
from skin_cancer.prep.structure_cnn import gradcam, lesion_crop, load_trained, predict

st.set_page_config(page_title="ABCD Teaching Demo", layout="wide")

DISCLAIMER = ("**Educational prototype — not clinically reviewed, not a medical device.** "
              "Do not use for decisions about real patients.")
SWATCH = {"white": (240, 240, 240), "red": (200, 40, 40), "light_brown": (181, 134, 84),
          "dark_brown": (101, 67, 33), "blue_gray": (100, 125, 160), "black": (20, 20, 20)}
LOW_AGREEMENT = {"red", "white"}
DX_NAME = {"mel": "Melanoma", "nv": "Melanocytic nevus (benign)", 0: "Common nevus (benign)",
           1: "Atypical nevus (benign)", 2: "Melanoma"}


# ----------------------------------------------------------------- resources

@st.cache_resource
def resources():
    ham = DATA_DIR / "ham10000"
    imgs = {}
    for i in (1, 2):
        z = zipfile.ZipFile(ham / f"HAM10000_images_part_{i}.zip")
        for n in z.namelist():
            if n.endswith(".jpg"):
                imgs[n.rsplit("/", 1)[-1][:-4]] = (z, n)
    zm = zipfile.ZipFile(ham / "HAM10000_segmentations_lesion_tschandl.zip")
    masks = {n.rsplit("/", 1)[-1].replace("_segmentation.png", ""): n
             for n in zm.namelist() if n.endswith(".png")}
    meta = pd.read_csv(ham / "HAM10000_metadata.tab", sep=None, engine="python").set_index("image_id")
    cnn, cnn_thr = load_trained()
    derm = joblib.load(DATA_DIR / "models" / "dermoscopic_models.joblib")
    ref = derm["train_reference"].reset_index(drop=True)
    X = ref[DERM_CONT].to_numpy(float)
    med = np.nanmedian(X, 0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clin = joblib.load(DATA_DIR / "models" / "clinical_models.joblib")
    return {"imgs": imgs, "zm": zm, "masks": masks, "meta": meta, "cnn": cnn, "cnn_thr": cnn_thr,
            "derm": derm, "ref": ref, "refX": (X - mu) / sd, "ref_stats": (med, mu, sd),
            "t": load_thresholds(), "clin": clin,
            "ph2": load_ph2().set_index("name"),
            "zp": zipfile.ZipFile(DATA_DIR / "ph2" / "PH2.zip"),
            "zi": zipfile.ZipFile(DATA_DIR / "isic2024" / "ISIC_2024_Training_Input.zip"),
            "ham_ho": pd.read_parquet(REPORTS / "stage4" / "ham10000_holdout_predictions.parquet"),
            "clin_ho": pd.read_parquet(REPORTS / "stage4" / "isic2024_holdout_predictions.parquet"),
            "clin_feat": pd.read_parquet(DATA_DIR / "prepared" / "isic2024_features.parquet",
                                         columns=["isic_id", "tbp_lv_symm_2axis", "tbp_lv_norm_border",
                                                  "tbp_lv_norm_color", "age_approx", "sex",
                                                  "anatom_site_general"]).set_index("isic_id"),
            "eval": json.loads((REPORTS / "stage5" / "evaluation.json").read_text()),
            "summary": json.loads((REPORTS / "stage4" / "modeling_summary.json").read_text())}


def ham_case(R, iid):
    z, n = R["imgs"][iid]
    rgb = np.array(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
    m = np.array(Image.open(io.BytesIO(R["zm"].read(R["masks"][iid]))).convert("L")) > 127
    return rgb, m


def ph2_case(R, name):
    zp = R["zp"]
    rgb = np.array(Image.open(io.BytesIO(zp.read(f"ph2_dataset/trainx/{name}.bmp"))).convert("RGB"))
    m = np.array(Image.open(io.BytesIO(zp.read(f"ph2_dataset/trainy/{name}_lesion.bmp"))).convert("L")) > 127
    return rgb, m


def auto_mask(rgb):
    """Fallback segmentation for uploads: Otsu on the blurred blue channel, central blob."""
    b = cv2.GaussianBlur(rgb[..., 2], (0, 0), 3)
    _, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lab, n = ndi.label(th > 0)
    if n == 0:
        return th > 0
    cy, cx = np.array(th.shape) // 2
    keep = lab[cy, cx] or (np.argmax(ndi.sum(th > 0, lab, range(1, n + 1))) + 1)
    return ndi.binary_fill_holes(lab == keep)


# ------------------------------------------------------------------ analysis

def analyse(R, rgb, mask, age, sex, site):
    res = extract(rgb, mask, R["t"], keep_evidence=True)
    crop = lesion_crop(rgb, mask)
    probs = predict(R["cnn"], crop[None])[0]
    names = ["pigment_network", "globules", "streaks", "negative_network", "milia_like_cyst"]
    p = dict(zip(names, probs))
    has = {k: p[k] >= R["cnn_thr"][k] for k in names}
    structless = res.features["d_has_structureless"] == 1
    D = max(1, int(has["pigment_network"]) + int(has["streaks"]) + int(has["globules"]) + int(structless))
    A, B, C = res.A, res.B, res.C
    tds = round(1.3 * A + 0.1 * B + 0.5 * C + 0.5 * D, 2)
    row = {**res.features, **{f"cnn_p_{k}": float(v) for k, v in p.items()},
           "age": age, "sex": sex, "localization": site, "A": A, "B": B, "C": C, "D": D}
    df = pd.DataFrame([row])
    model = R["derm"]["models"]["D3c"]
    raw = float(model.predict_proba(df)[0, 1])
    prior = R["summary"]["dermoscopic"]["mel_train"] / R["summary"]["dermoscopic"]["n_train"]
    logit = np.log(raw / (1 - raw)) + np.log(prior / (1 - prior))
    prob = float(1 / (1 + np.exp(-logit)))
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    xt = pre.transform(df)
    xt = xt.toarray()[0] if hasattr(xt, "toarray") else np.asarray(xt)[0]
    contrib = pd.Series(xt * clf.coef_[0], index=pre.get_feature_names_out())
    # similar training cases (standardised continuous ABCD space)
    med, mu, sd = R["ref_stats"]
    v = df[DERM_CONT].to_numpy(float)[0]
    v = (np.where(np.isnan(v), med, v) - mu) / sd
    d = np.linalg.norm(R["refX"] - v, axis=1)
    nn_idx = np.argsort(d)[:5]
    return {"res": res, "crop": crop, "p": p, "has": has, "structless": structless,
            "A": A, "B": B, "C": C, "D": D, "TDS": tds, "prob": prob, "contrib": contrib,
            "neighbours": R["ref"].iloc[nn_idx].assign(distance=d[nn_idx])}


# ------------------------------------------------------------------ drawing

def draw_outline(rgb, mask, color=(0, 255, 0)):
    out = rgb.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, color, 2)
    return out


def draw_A(ev):
    rgb, mask, a = ev["rgb"], ev["mask"], ev["A"]
    out = draw_outline(rgb, mask, (255, 255, 255))
    cx, cy = a["center_xy"]
    th = np.radians(a["axis_angle_deg"])
    r = 1.3 * np.sqrt(mask.sum() / np.pi)
    for d, asym in [((np.cos(th), -np.sin(th)), a["axis_asymmetric"][0]),
                    ((np.sin(th), np.cos(th)), a["axis_asymmetric"][1])]:
        p1 = (int(cx - r * d[0]), int(cy - r * d[1]))
        p2 = (int(cx + r * d[0]), int(cy + r * d[1]))
        cv2.line(out, p1, p2, (255, 40, 40) if asym else (40, 220, 40), 3)
    return out


def draw_B(ev):
    rgb, mask, b = ev["rgb"], ev["mask"], ev["B"]
    out = rgb.copy()
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ang = ev["A"]["axis_angle_deg"]
    for c in cs:
        for (x, y) in c[:, 0, :]:
            theta = (np.degrees(np.arctan2(y - cy, x - cx)) - ang) % 360
            s = int(theta // 45)
            cv2.circle(out, (int(x), int(y)), 2, (255, 30, 30) if b["sector_abrupt"][s] else (255, 220, 0), -1)
    for k in range(8):
        t = np.radians(ang + 45 * k)
        r = 1.2 * np.sqrt(mask.sum() / np.pi)
        cv2.line(out, (int(cx), int(cy)), (int(cx + r * np.cos(t)), int(cy + r * np.sin(t))), (200, 200, 200), 1)
    return out


def draw_C(ev):
    rgb, lm = ev["rgb"], ev["C"]["label_map"]
    over = rgb.copy()
    for i, n in enumerate(COLOR_NAMES):
        over[lm == i] = SWATCH[n]
    return cv2.addWeighted(rgb, 0.35, over, 0.65, 0)


def heat(crop, cam):
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return cv2.addWeighted(crop, 0.55, hm, 0.45, 0)


# ------------------------------------------------------------------ pages

def page_case(R):
    st.header("Dermoscopic case walk-through — how an ABCD assessment is built")
    src = st.radio("Case source", ["Held-out HAM10000 case (never used for training)",
                                   "PH2 case (external centre, expert labels)", "Upload your own image"],
                   horizontal=True)
    truth, expert = None, None
    if src.startswith("Held-out"):
        ho = R["ham_ho"]
        if "case_idx" not in st.session_state:
            st.session_state.case_idx = 0
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("🎲 Random case"):
                st.session_state.case_idx = int(np.random.randint(len(ho)))
                st.session_state.pop("analysis", None)
        with c1:
            idx = st.selectbox("Case", range(len(ho)), index=st.session_state.case_idx,
                               format_func=lambda i: f"Case {i + 1:03d} · {ho.image_id.iloc[i]}")
        iid = ho.image_id.iloc[idx]
        rgb, mask = ham_case(R, iid)
        m = R["meta"].loc[iid]
        age = float(m.age) if pd.notna(m.age) else np.nan
        sex, site = m.sex, m.localization
        truth = DX_NAME[m.dx]
        key = iid
    elif src.startswith("PH2"):
        names = list(R["ph2"].index)
        name = st.selectbox("PH2 image", names)
        rgb, mask = ph2_case(R, name)
        age, sex, site = np.nan, "unknown", "unknown"
        e = R["ph2"].loc[name]
        truth = DX_NAME[int(e.clinical_dx)]
        expert = {"A": int(e.asymmetry), "C": len(e.colors.split()),
                  "colours": [COLOR_NAMES[int(c) - 1] for c in e.colors.split()]}
        key = name
    else:
        up = st.file_uploader("Dermoscopic image (jpg/png)", type=["jpg", "jpeg", "png", "bmp"])
        if not up:
            st.info("Upload a dermoscopic image centred on the lesion.")
            return
        rgb = np.array(Image.open(up).convert("RGB"))
        mask = auto_mask(rgb)
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 0, 100, 50)
        sex = c2.selectbox("Sex", ["male", "female", "unknown"])
        site = c3.selectbox("Site", ["back", "trunk", "lower extremity", "upper extremity", "face",
                                     "abdomen", "chest", "unknown"])
        key = up.name

    left, right = st.columns([1, 1])
    with left:
        st.image(rgb, caption="Input image", width="stretch")
    with right:
        st.subheader("Step 0 — Score it yourself first")
        with st.form(f"student_{key}"):
            sa = st.slider("A — asymmetry axes (0–2)", 0, 2, 0)
            sb = st.slider("B — abrupt border segments (0–8)", 0, 8, 0)
            sc = st.slider("C — number of colours (1–6)", 1, 6, 1)
            sd = st.slider("D — differential structures (1–5)", 1, 5, 1)
            go = st.form_submit_button("Submit my scores and show the analysis")
        if go:
            st.session_state.student = {"A": sa, "B": sb, "C": sc, "D": sd}
            with st.spinner("Running ABCD extraction + structure detectors…"):
                st.session_state.analysis = (key, analyse(R, rgb, mask, age, sex, site))
    if "analysis" not in st.session_state or st.session_state.analysis[0] != key:
        return
    a = st.session_state.analysis[1]
    ev, s = a["res"].evidence, st.session_state.student

    st.divider()
    st.subheader("Step 1 — Find the lesion")
    st.image(draw_outline(ev["rgb"], ev["mask"]), width=420,
             caption="Lesion outline used for every measurement (hair removed, skin-colour normalised)")

    st.subheader("Step 2 — Score each criterion, with the evidence")
    cA, cB, cC = st.columns(3)
    fa = a["res"].features
    with cA:
        st.markdown(f"**A = {a['A']}** · asymmetry")
        st.image(draw_A(ev), width="stretch")
        st.caption(f"Best axis pair; red = asymmetric axis. Contour mismatch "
                   f"{fa['a_shape_ax1']:.2f} / {fa['a_shape_ax2']:.2f} (cut 0.225); half-colour ΔE "
                   f"{fa['a_color_ax1']:.1f} / {fa['a_color_ax2']:.1f} (cut 10). Agreement with experts on PH2: κ 0.40.")
    with cB:
        st.markdown(f"**B = {a['B']}** · abrupt border sectors")
        st.image(draw_B(ev), width="stretch")
        st.caption("8 sectors; red dots = abrupt pigment cut-off, yellow = gradual. "
                   "⚠️ No public ground truth for B — heuristic, under-counts vs experts.")
    with cC:
        st.markdown(f"**C = {a['C']}** · colours")
        st.image(draw_C(ev), width="stretch")
        cols = ev["C"]["colors_present"]
        st.caption("Colours counted: " + (", ".join(
            f"{c.replace('_', ' ')}{' ⚠️' if c in LOW_AGREEMENT else ''}" for c in cols) or "—")
                   + ". ⚠️ = low agreement with experts (red, white). Colour count κ vs experts is weak.")
    st.markdown(f"**D = {a['D']}** · differential structures (learned detectors, Grad-CAM shows *where*)")
    dcols = st.columns(4)
    for col, k, label in zip(dcols, ["pigment_network", "globules", "streaks"],
                             ["Pigment network", "Dots / globules", "Streaks"]):
        with col:
            st.image(heat(a["crop"], gradcam(R["cnn"], a["crop"], k)), width="stretch")
            st.caption(f"{label}: p = {a['p'][k]:.2f} → **{'present' if a['has'][k] else 'absent'}**")
    with dcols[3]:
        st.image(np.where(ev["D"]["structureless_mask"][..., None], np.array([255, 0, 255], np.uint8),
                          ev["rgb"]).astype(np.uint8), width="stretch")
        st.caption(f"Structureless area (magenta): **{'present' if a['structless'] else 'absent'}**")

    st.subheader("Step 3 — The Stolz rule (Total Dermoscopy Score)")
    st.latex(rf"TDS = 1.3\times{a['A']} + 0.1\times{a['B']} + 0.5\times{a['C']} + 0.5\times{a['D']} = \mathbf{{{a['TDS']}}}")
    verdict = "benign" if a["TDS"] < 4.75 else ("suspicious" if a["TDS"] <= 5.45 else "highly suspicious for melanoma")
    cut = R["summary"]["dermoscopic"]["tds_recalibrated_cut_sens85_train"]
    st.markdown(f"- Published cut-offs: < 4.75 benign · 4.75–5.45 suspicious · > 5.45 melanoma → **{verdict}**\n"
                f"- ⚠️ **What this project found:** with *automated* scoring the published cut-offs catch only "
                f"8% of melanomas (published: 85%). A cut-off of **{cut}** catches 88% but flags 77% of nevi. "
                f"At {cut} this case would be **{'flagged' if a['TDS'] >= cut else 'not flagged'}**.")

    st.subheader("Step 4 — What the data say about the same measurements")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Melanoma probability (interpretable model)", f"{a['prob']:.0%}")
        st.caption("Logistic regression on the continuous ABCD measurements + age/sex/site; holdout AUC 0.85, "
                   "external PH2 AUC 0.83. Calibrated for a referral population with ~31% melanoma.")
    with c2:
        top = a["contrib"].reindex(a["contrib"].abs().sort_values(ascending=False).index)[:8]
        lab = [n.replace("num__", "").replace("cat__", "") for n in top.index]
        st.bar_chart(pd.DataFrame({"contribution to log-odds": top.values}, index=lab), horizontal=True)
        st.caption("Positive bars push towards melanoma, negative towards benign.")

    st.subheader("Step 5 — Similar confirmed cases")
    nb = st.columns(5)
    for col, (_, r) in zip(nb, a["neighbours"].iterrows()):
        with col:
            nrgb, nm = ham_case(R, r.image_id)
            st.image(lesion_crop(nrgb, nm, size=200), width="stretch")
            st.caption(f"{DX_NAME[r.dx]} · TDS {r.TDS}")

    st.subheader("Step 6 — Known blind spots")
    warn = []
    if a["A"] == 0:
        warn.append("Lesion scored **symmetric**: missed melanomas in our holdout were mostly symmetric (mean A 0.09).")
    if fa["lesion_area_frac"] < 0.05:
        warn.append("Lesion is **small** in the image: small melanomas are often missed by ABCD.")
    warn.append("ABCD is designed for pigmented melanocytic lesions; amelanotic/nodular melanomas and non-melanoma "
                "cancers (BCC, SCC) are outside its scope.")
    for w in warn:
        st.warning(w)

    st.subheader("Step 7 — Reveal")
    if st.button("Show the confirmed diagnosis"):
        st.success(f"Confirmed diagnosis: **{truth}**" if truth else "No ground truth for uploads.")
        comp = pd.DataFrame({"You": s, "System": {k: a[k] for k in "ABCD"}})
        if expert:
            comp["PH2 expert"] = {"A": expert["A"], "B": None, "C": expert["C"], "D": None}
        st.table(comp)
        if expert:
            st.caption("Expert colours: " + ", ".join(expert["colours"]))


def page_clinical(R):
    st.header("Clinical ABCD(E) on total-body photography (ISIC 2024)")
    ho, feat, flags = R["clin_ho"], R["clin_feat"], R["clin"]["count_rule_flags"]
    pick = st.radio("Show a", ["random benign crop", "random melanoma crop"], horizontal=True)
    if st.button("🎲 New crop") or "clin_id" not in st.session_state or st.session_state.get("clin_pick") != pick:
        pool = ho[ho.y_melanoma == (1 if "melanoma" in pick else 0)]
        st.session_state.clin_id = pool.isic_id.iloc[np.random.randint(len(pool))]
        st.session_state.clin_pick = pick
    iid = st.session_state.clin_id
    r, f = ho.set_index("isic_id").loc[iid], feat.loc[iid]
    img = Image.open(io.BytesIO(R["zi"].read(f"ISIC_2024_Training_Input/{iid}.jpg")))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, width=260, caption=f"{iid} · 15×15 mm TBP crop")
    with c2:
        rows = [("A asymmetry index", f.tbp_lv_symm_2axis, flags["A"][1]),
                ("B border irregularity", f.tbp_lv_norm_border, flags["B"][1]),
                ("C colour variation", f.tbp_lv_norm_color, flags["C"][1]),
                ("D long diameter (mm)", r.clin_size_long_diam_mm, 6.0)]
        tbl = pd.DataFrame([{"criterion": n, "value": round(float(v), 2), "flag above": round(t, 2),
                             "positive": "✅" if v > t else "—"} for n, v, t in rows])
        st.table(tbl)
        n_pos = int(r.count_rule)
        st.markdown(f"**ABCD count = {n_pos} / 4** → {'**suspicious** (≥ 2)' if n_pos >= 2 else 'not flagged'}")
        e = R["eval"]["clinical_holdout"]
        st.info(f"On 100,362 held-out crops, *diameter > 6 mm* flags {e['D>6mm_operating_point']['flagged_per_100k_crops']:,} "
                f"per 100,000 lesions — about **{e['D>6mm_operating_point']['NNE_at_holdout_prevalence']:.0f} excisions per "
                f"melanoma found**. ABCD is for lesions that already stand out, not for screening every mole.")
        if st.button("Reveal diagnosis"):
            st.success("Melanoma" if r.y_melanoma == 1 else "Benign (clinically assessed)")


def page_results(R):
    st.header("What the project found")
    e = R["eval"]
    d = e["dermoscopic_holdout"]
    rows = [(k, v["auc"]) for k, v in d.items() if isinstance(v, dict) and "auc" in v]
    st.subheader("Dermoscopic holdout (496 lesions, 154 melanomas) — AUC [95% CI]")
    st.table(pd.DataFrame([{"model": k, "AUC": a[0], "CI low": a[1], "CI high": a[2]} for k, a in rows]))
    s1 = d["S1_rule_confirmation"]
    st.subheader("Does the published Stolz rule hold with automated scoring?")
    st.table(pd.DataFrame([{"rule": k, "sensitivity": v["sens"][0], "specificity": v["spec"][0]}
                           for k, v in s1.items() if "harrington" not in k]
                          + [{"rule": "Published pooled (Harrington 2017, TDS>4.75)", "sensitivity": 0.85,
                              "specificity": 0.72}]))
    st.markdown("Full reports: `crisp/01`–`06`, `docs/REPORT.md`.")


def main():
    st.sidebar.title("ABCD Teaching Demo")
    st.sidebar.warning(DISCLAIMER)
    page = st.sidebar.radio("Page", ["Dermoscopic case walk-through", "Clinical TBP panel", "Project results"])
    st.sidebar.caption("Data: HAM10000, ISIC 2018/2024 (CC-BY-NC), PH2 (research/education, local only).")
    R = resources()
    {"Dermoscopic case walk-through": page_case, "Clinical TBP panel": page_clinical,
     "Project results": page_results}[page](R)


main()
