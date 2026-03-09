"""
LeafLab — Full Morphology Pipeline
Steps 1-9 as specified.
"""
import time
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ─── helpers ────────────────────────────────────────────────────────────────

def _encode_img(arr: np.ndarray, ext: str = ".png") -> str:
    """Encode numpy BGR/gray image to base64 data URI."""
    import base64
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf).decode()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


# ─── Step 1: Preprocessing ──────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray):
    """Resize → GaussianBlur → CLAHE on L channel."""
    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(img_bgr, (3, 3), 0)

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return enhanced, rgb, lab_eq


# ─── Step 2: Segmentation ───────────────────────────────────────────────────

def segment(rgb: np.ndarray, use_watershed: bool = False):
    from utils.segmentation import segment_leaf
    return segment_leaf(rgb, use_watershed)


# ─── Step 3: Morphological Refinement ───────────────────────────────────────

def morph_refine(mask: np.ndarray, se_shape: str = "ellipse", se_size: int = 9) -> dict:
    from utils.segmentation import apply_morphology
    return apply_morphology(mask, se_shape, se_size)


# ─── Step 4: Skeletonization ────────────────────────────────────────────────

def skeletonize_mask(mask: np.ndarray):
    """Return skeleton bool array, length, branch_points count."""
    try:
        from skimage.morphology import skeletonize as ski_skel
        binary = (mask > 0)
        skel = ski_skel(binary)
    except Exception:
        # fallback: Zhang-Suen via OpenCV thinning
        m = (mask > 0).astype(np.uint8)
        skel_img = np.zeros_like(m, dtype=bool)
        temp = m.copy()
        while True:
            eroded = cv2.erode(temp, np.ones((3, 3), np.uint8))
            opened = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
            subset = temp - opened
            skel_img |= (subset > 0)
            temp = eroded
            if cv2.countNonZero(temp) == 0:
                break
        skel = skel_img

    skeleton_length = int(skel.sum())

    # Count branch points (pixels with >2 skeleton neighbours)
    skel_uint = skel.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skel_uint.astype(np.float32), -1,
                                   kernel.astype(np.float32)) - skel_uint
    branch_points = int(((neighbor_count > 2) & skel).sum())

    return skel, skeleton_length, branch_points


# ─── Step 5: Convex Hull ────────────────────────────────────────────────────

def compute_hull(mask: np.ndarray, img_bgr: np.ndarray):
    """Return hull_image (BGR), hull_area, solidity."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull_img = img_bgr.copy()
    if not contours:
        return hull_img, 0.0, 0.0

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    leaf_area = cv2.contourArea(cnt)
    solidity = leaf_area / (hull_area + 1e-6)

    cv2.drawContours(hull_img, [hull], -1, (45, 106, 79), 2)
    cv2.drawContours(hull_img, [cnt], -1, (82, 183, 136), 1)

    return hull_img, float(hull_area), float(solidity)


# ─── Step 6: Feature Extraction (22 features) ───────────────────────────────

def extract_features(mask: np.ndarray, rgb: np.ndarray,
                     skeleton_length: int, branch_points: int) -> dict:
    """Extract 22 morphological + texture + color features."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _zero_features()

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / (h + 1e-6)
    extent = area / (w * h + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / (hull_area + 1e-6)

    equiv_diameter = float(np.sqrt(4 * area / np.pi))

    ellipse = None
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        ma, MA = ellipse[1]
        eccentricity = float(np.sqrt(1 - (min(ma, MA) / (max(ma, MA) + 1e-6)) ** 2))
    else:
        eccentricity = 0.0

    # Hu Moments
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # Texture — GLCM approximation via gray stats
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    leaf_pixels = gray[binary == 1]
    if len(leaf_pixels) > 10:
        glcm_contrast   = float(np.var(leaf_pixels.astype(np.float32)))
        glcm_energy     = float(np.sum((leaf_pixels.astype(np.float32) / 255.0) ** 2) /
                                (len(leaf_pixels) + 1e-6))
        glcm_homogeneity = float(1.0 / (1.0 + np.var(leaf_pixels.astype(np.float32)) / 1000.0))
        # correlation proxy
        if len(leaf_pixels) > 1:
            shifted = np.roll(leaf_pixels, 1)
            glcm_correlation = float(np.corrcoef(leaf_pixels.astype(np.float32),
                                                  shifted.astype(np.float32))[0, 1])
            if np.isnan(glcm_correlation):
                glcm_correlation = 0.0
        else:
            glcm_correlation = 0.0
    else:
        glcm_contrast = glcm_energy = glcm_homogeneity = glcm_correlation = 0.0

    # Color — ExG, Hue, Saturation over leaf pixels
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    mask_bool = binary == 1

    mean_exg = float(exg[mask_bool].mean()) if mask_bool.any() else 0.0

    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    mean_hue = float(hsv[:, :, 0][mask_bool].mean()) if mask_bool.any() else 0.0
    mean_saturation = float(hsv[:, :, 1][mask_bool].mean()) if mask_bool.any() else 0.0

    return {
        "area":             area,
        "perimeter":        perimeter,
        "circularity":      float(np.clip(circularity, 0, 1)),
        "aspect_ratio":     float(np.clip(aspect_ratio, 0, 20)),
        "solidity":         float(np.clip(solidity, 0, 1)),
        "extent":           float(np.clip(extent, 0, 1)),
        "equiv_diameter":   equiv_diameter,
        "eccentricity":     float(np.clip(eccentricity, 0, 1)),
        "hu_1":             float(hu_log[0]),
        "hu_2":             float(hu_log[1]),
        "hu_3":             float(hu_log[2]),
        "hu_4":             float(hu_log[3]),
        "hu_5":             float(hu_log[4]),
        "hu_6":             float(hu_log[5]),
        "hu_7":             float(hu_log[6]),
        "glcm_contrast":    glcm_contrast,
        "glcm_energy":      glcm_energy,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_correlation": glcm_correlation,
        "mean_exg":         mean_exg,
        "mean_hue":         mean_hue,
        "skeleton_length":  float(skeleton_length),
        "branch_points":    float(branch_points),
        "mean_saturation":  mean_saturation,
        "pct_vegetation":   float(100.0 * mask_bool.sum() / (mask.shape[0] * mask.shape[1])),
    }


def _zero_features() -> dict:
    keys = [
        "area","perimeter","circularity","aspect_ratio","solidity",
        "extent","equiv_diameter","eccentricity",
        "hu_1","hu_2","hu_3","hu_4","hu_5","hu_6","hu_7",
        "glcm_contrast","glcm_energy","glcm_homogeneity","glcm_correlation",
        "mean_exg","mean_hue","skeleton_length","branch_points",
        "mean_saturation","pct_vegetation"
    ]
    return {k: 0.0 for k in keys}


# ─── Step 8: Rule-based Classification ─────────────────────────────────────

def rule_classify(features: dict):
    """
    Rule-based classifier — returns (species_label, confidence).
    species_label includes both shape class and botanical hint.
    """
    c  = features.get("circularity", 0)
    s  = features.get("solidity", 0)
    ar = features.get("aspect_ratio", 1)
    bp = features.get("branch_points", 0)
    sl = features.get("skeleton_length", 0)

    if c > 0.65 and s > 0.85 and ar < 1.8:
        return "Healthy Broad Leaf", 0.82
    elif ar > 2.5 or (c < 0.42 and s > 0.78):
        return "Elongated Narrow Leaf", 0.78
    elif s < 0.72 or bp > 20:
        return "Lobed / Serrated Leaf", 0.74
    elif c > 0.50 and ar < 1.3:
        return "Ovate / Round Leaf", 0.76
    else:
        return "Irregular / Compound Leaf", 0.70


SPECIES_DETAIL = {
    "Healthy Broad Leaf":       {"shape": "Broad / Oval",      "family": "Dicotyledon",  "example": "e.g. Mango, Ficus, Teak"},
    "Elongated Narrow Leaf":    {"shape": "Linear / Lanceolate","family": "Monocotyledon","example": "e.g. Grass, Bamboo, Sugarcane"},
    "Lobed / Serrated Leaf":    {"shape": "Lobed / Palmate",   "family": "Dicotyledon",  "example": "e.g. Maple, Oak, Papaya"},
    "Ovate / Round Leaf":       {"shape": "Ovate / Cordate",   "family": "Dicotyledon",  "example": "e.g. Betel, Peepal, Lotus"},
    "Irregular / Compound Leaf":{"shape": "Compound / Pinnate","family": "Mixed",         "example": "e.g. Neem, Tamarind, Rose"},
}


# ─── Step 9: Full Pipeline ──────────────────────────────────────────────────

def run_pipeline(img_bgr: np.ndarray, se_shape: str = "ellipse",
                 se_size: int = 9, use_watershed: bool = False,
                 location: str = "") -> dict:
    """
    End-to-end pipeline. Returns complete result dict.
    """
    t0 = time.time()

    # 1. Preprocess
    img_bgr, rgb, lab = preprocess(img_bgr)

    # 2. Segment
    leaf_mask, dry_mask = segment(rgb, use_watershed)

    # 3. Morph refine
    morph = morph_refine(leaf_mask, se_shape, se_size)
    final_mask = morph["final"]

    # 4. Skeletonize
    skeleton, skel_len, branch_pts = skeletonize_mask(final_mask)

    # 5. Convex Hull
    hull_img, hull_area, solidity_hull = compute_hull(final_mask, img_bgr)

    # 6. Features
    features = extract_features(final_mask, rgb, skel_len, branch_pts)
    features["solidity"] = max(features.get("solidity", 0), solidity_hull * 0.5)

    # 7. (Watershed already handled in step 2)

    # 8. Rule classify
    species, confidence = rule_classify(features)

    # Try ML model
    ml_species, ml_conf = _try_ml_predict(features)
    if ml_species:
        # Ensemble: average confidence
        species = ml_species
        confidence = round((confidence + ml_conf) / 2, 3)
        method = "Ensemble ML"
    else:
        method = "Rule-based"

    process_time = round(time.time() - t0, 3)
    features["process_time"] = process_time

    # Build skeleton overlay image — overlay ONLY on the leaf mask region
    skel_vis = img_bgr.copy()
    # Dim the background slightly so skeleton stands out
    bg_mask = (final_mask == 0)
    skel_vis[bg_mask] = (skel_vis[bg_mask].astype(np.float32) * 0.5).astype(np.uint8)
    # Draw skeleton in bright green
    skel_vis[skeleton] = [0, 230, 80]  # BGR green

    # Contour overlay
    contour_vis = img_bgr.copy()
    cnts, _ = cv2.findContours((final_mask > 0).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_vis, cnts, -1, (45, 106, 79), 2)

    # ExG heatmap
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b_ch = rgb[:, :, 2].astype(np.float32)
    exg_map = 2 * g - r - b_ch
    exg_norm = cv2.normalize(exg_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(exg_norm, cv2.COLORMAP_JET)

    # Recommendation
    recommendation = _build_recommendation(species, features)

    images = {
        "raw":         _encode_img(img_bgr),
        "segmented":   _encode_img(_mask_to_bgr(final_mask)),
        "eroded":      _encode_img(_mask_to_bgr(morph["eroded"])),
        "dilated":     _encode_img(_mask_to_bgr(morph["dilated"])),
        "opened":      _encode_img(_mask_to_bgr(morph["opened"])),
        "closed":      _encode_img(_mask_to_bgr(morph["closed"])),
        "skeleton":    _encode_img(skel_vis),
        "hull":        _encode_img(hull_img),
        "heatmap":     _encode_img(heatmap),
        "contour":     _encode_img(contour_vis),
    }

    return {
        "species":          species,
        "confidence":       confidence,
        "method":           method,
        "features":         features,
        "images":           images,
        "skeleton_stats":   {"length": skel_len, "branch_points": branch_pts},
        "hull_stats":       {"hull_area": hull_area, "solidity": solidity_hull},
        "recommendation":   recommendation,
        "process_time":     process_time,
        "location":         location,
        "species_detail":   SPECIES_DETAIL.get(species, {"shape": "—", "family": "—", "example": "—"}),
    }


def _try_ml_predict(features: dict):
    """Try to load trained model and predict. Returns (species, conf) or (None, 0)."""
    import os
    import pickle
    model_path  = "models/ensemble_model.pkl"
    scaler_path = "models/scaler.pkl"
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, 0.0
    try:
        from utils.trainer import predict_leaf
        return predict_leaf(features, model_path, scaler_path)
    except Exception as e:
        logger.warning(f"ML predict failed: {e}")
        return None, 0.0


def _build_recommendation(species: str, features: dict) -> dict:
    pct = features.get("pct_vegetation", 0)
    circ = features.get("circularity", 0)
    sol = features.get("solidity", 0)

    if circ > 0.6 and sol > 0.8:
        level = "healthy"
        title = "Specimen Appears Healthy"
        desc  = "Morphology indicates a well-formed, intact leaf structure with good tissue density."
    elif sol > 0.65:
        level = "warning"
        title = "Minor Morphological Irregularities"
        desc  = "Some deviation in shape metrics detected. Consider closer examination of margins."
    else:
        level = "critical"
        title = "Significant Morphological Abnormality"
        desc  = "Shape descriptors indicate possible structural damage, disease, or mechanical stress."

    return {"level": level, "title": title, "desc": desc, "species": species}
