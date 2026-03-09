"""
LeafLab — LIME Explainer with ExG fallback saliency.
Never raises exceptions to caller.
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _encode_img(arr: np.ndarray) -> str:
    import base64
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def explain_prediction(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """
    Generate LIME explanation or fallback ExG saliency.
    Always returns a dict with 'original' and 'explanation' base64 images.
    Never raises.
    """
    try:
        return _lime_explain(img_bgr, mask)
    except Exception as e:
        logger.warning(f"LIME failed ({e}), using fallback saliency")
        return _fallback_saliency(img_bgr, mask)


def _lime_explain(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Attempt LIME superpixel explanation."""
    from lime import lime_image
    from skimage.segmentation import slic
    from sklearn.linear_model import Ridge
    import functools

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Simple classifier proxy: predict based on ExG in segment
    def _classifier(imgs):
        results = []
        for im in imgs:
            r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
            exg = 2.0 * g - r.astype(float) - b.astype(float)
            score = float(np.clip(exg.mean() / 50.0, 0, 1))
            results.append([1 - score, score])
        return np.array(results)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        rgb,
        _classifier,
        top_labels=1,
        hide_color=0,
        num_samples=50,
    )

    top_label = explanation.top_labels[0]
    temp, lime_mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )

    # Create overlay
    overlay = rgb.copy()
    overlay[lime_mask == 1] = np.clip(
        overlay[lime_mask == 1].astype(int) + [0, 60, 0], 0, 255
    ).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    return {
        "original":    _encode_img(img_bgr),
        "explanation": _encode_img(overlay_bgr),
        "method":      "LIME",
    }


def _fallback_saliency(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """ExG heatmap overlay as fallback saliency map."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    exg = 2 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Gaussian blur for smooth saliency
    exg_smooth = cv2.GaussianBlur(exg_norm, (21, 21), 0)
    heatmap = cv2.applyColorMap(exg_smooth, cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

    # Apply mask if available
    if mask is not None and mask.any():
        bin_mask = (mask > 0).astype(np.uint8)
        bin_mask_3ch = cv2.merge([bin_mask, bin_mask, bin_mask]) * 255
        overlay = cv2.bitwise_and(overlay, bin_mask_3ch) + \
                  cv2.bitwise_and(img_bgr, cv2.bitwise_not(bin_mask_3ch))

    return {
        "original":    _encode_img(img_bgr),
        "explanation": _encode_img(overlay),
        "method":      "ExG Saliency (fallback)",
    }
