"""
LeafLab — Segmentation utilities
ExG-based leaf segmentation, mask cleanup, watershed.
"""
import numpy as np
import cv2


def compute_exg(rgb: np.ndarray) -> np.ndarray:
    """Excess Green Index: 2G - R - B (float32)."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return 2.0 * g - r - b


def compute_pseudo_ndvi(rgb: np.ndarray) -> np.ndarray:
    """Pseudo-NDVI using green/red channels."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    denom = r + g + 1e-6
    return (g - r) / denom


def segment_leaf(rgb: np.ndarray, use_watershed: bool = False):
    """
    Segment leaf from background using a multi-cue approach:
    ExG + LAB + HSV saturation — works for green, yellow, and dry leaves.
    Returns: leaf_mask (uint8, 0/255), dry_mask (uint8, 0/255)
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── Cue 1: ExG (works well for green leaves) ──────────────────────────
    exg = compute_exg(rgb)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_exg = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Cue 2: LAB b* channel — yellow/green vs blue background ──────────
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b_ch = lab[:, :, 2]  # b* > 128 = yellow/green, < 128 = blue
    # Leaves (even yellow ones) have higher b* than grey/white/blue backgrounds
    b_norm = cv2.normalize(b_ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_lab = cv2.threshold(b_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Cue 3: HSV saturation — leaves are more saturated than white/grey bg ─
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask_sat = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)

    # ── Combine: leaf must pass LAB b* AND (ExG OR saturation) ───────────
    mask_combined = cv2.bitwise_and(mask_lab,
                    cv2.bitwise_or(mask_exg, mask_sat))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN,  kernel, iterations=2)
    leaf_mask = cv2.morphologyEx(leaf_mask,     cv2.MORPH_CLOSE, kernel, iterations=3)

    # Keep only largest component
    leaf_mask = _keep_largest_component(leaf_mask)

    # Fill holes
    leaf_mask = _fill_holes(leaf_mask)

    # If multi-cue mask is too small (< 5% of image), fall back to ExG alone
    total_px = leaf_mask.shape[0] * leaf_mask.shape[1]
    if leaf_mask.sum() / 255 < 0.05 * total_px:
        leaf_mask_fallback = cv2.morphologyEx(mask_exg, cv2.MORPH_OPEN,  kernel, iterations=2)
        leaf_mask_fallback = cv2.morphologyEx(leaf_mask_fallback, cv2.MORPH_CLOSE, kernel, iterations=3)
        leaf_mask_fallback = _keep_largest_component(leaf_mask_fallback)
        leaf_mask_fallback = _fill_holes(leaf_mask_fallback)
        if leaf_mask_fallback.sum() > leaf_mask.sum():
            leaf_mask = leaf_mask_fallback

    # Dry region mask (yellowish areas within the leaf)
    pseudo_ndvi = compute_pseudo_ndvi(rgb)
    dry = ((exg < 0) & (pseudo_ndvi < 0.05)).astype(np.uint8) * 255
    dry_mask = cv2.bitwise_and(dry, leaf_mask)

    if use_watershed and leaf_mask.sum() > 0:
        leaf_mask = _apply_watershed(rgb, leaf_mask)

    return leaf_mask, dry_mask


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Retain only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood-fill from border to identify holes."""
    flood = mask.copy()
    h, w = flood.shape
    flood_fill = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_fill, (0, 0), 255)
    inverted = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inverted)


def _apply_watershed(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Refine mask using distance-transform watershed."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)

    out = np.zeros_like(mask)
    out[markers > 1] = 255
    return out


def apply_morphology(mask: np.ndarray, se_shape: str = "ellipse",
                     se_size: int = 9) -> dict:
    """
    Apply erosion/dilation/opening/closing on mask.
    Returns dict of intermediate and final masks.
    """
    se_size = max(3, se_size | 1)  # ensure odd

    if se_shape == "rect":
        shape = cv2.MORPH_RECT
    elif se_shape == "cross":
        shape = cv2.MORPH_CROSS
    else:
        shape = cv2.MORPH_ELLIPSE

    kernel = cv2.getStructuringElement(shape, (se_size, se_size))

    eroded  = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    opened  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    closed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Final: open then close on original
    refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)
    filled  = _fill_holes(refined)

    return {
        "eroded":  eroded,
        "dilated": dilated,
        "opened":  opened,
        "closed":  closed,
        "filled":  filled,
        "final":   filled,
    }
