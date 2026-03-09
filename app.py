"""
LeafLab — Flask Application
"""
import os
import io
import base64
import logging
import time
import numpy as np
import cv2

from flask import Flask, request, jsonify, render_template, send_file, Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# In-memory result cache for PDF generation
_last_result = {}


# ─── helpers ────────────────────────────────────────────────────────────────

def _read_image_from_request(field="image"):
    """Read image bytes from multipart or base64 JSON."""
    if field in request.files:
        f = request.files[field]
        data = f.read()
    else:
        data_b64 = request.json.get("image_b64", "")
        if "," in data_b64:
            data_b64 = data_b64.split(",", 1)[1]
        data = base64.b64decode(data_b64)

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    global _last_result
    try:
        img = _read_image_from_request("image")
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        se_shape    = request.form.get("se_shape", "ellipse")
        se_size     = int(request.form.get("se_size", 9))
        use_ws      = request.form.get("watershed", "false").lower() == "true"
        location    = request.form.get("location", "")
        filename    = request.files.get("image", type(lambda x: x)).__class__.__name__

        from utils.pipeline import run_pipeline
        result = run_pipeline(img, se_shape=se_shape, se_size=se_size,
                               use_watershed=use_ws, location=location)

        # Log to DB
        try:
            from utils.database import log_analysis
            fname = ""
            if "image" in request.files:
                fname = request.files["image"].filename or ""
            log_analysis(fname, result["species"], result["confidence"],
                         result["features"], location)
        except Exception as e:
            logger.warning(f"DB log failed: {e}")

        _last_result = result
        return jsonify(result)

    except Exception as e:
        logger.error(f"/api/analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_analyze", methods=["POST"])
def batch_analyze():
    from utils.pipeline import run_pipeline
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    se_shape = request.form.get("se_shape", "ellipse")
    se_size  = int(request.form.get("se_size", 9))
    use_ws   = request.form.get("watershed", "false").lower() == "true"

    results       = []
    total_processed = 0
    total_failed    = 0

    for f in files:
        try:
            data = f.read()
            arr  = np.frombuffer(data, dtype=np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                total_failed += 1
                results.append({"filename": f.filename, "error": "decode failed"})
                continue

            r = run_pipeline(img, se_shape=se_shape, se_size=se_size, use_watershed=use_ws)

            try:
                from utils.database import log_analysis
                log_analysis(f.filename, r["species"], r["confidence"], r["features"])
            except Exception:
                pass

            results.append({
                "filename":    f.filename,
                "species":     r["species"],
                "confidence":  r["confidence"],
                "method":      r["method"],
                "features":    {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in r["features"].items()},
                "images":      {k: v for k, v in r["images"].items()
                                if k in ("raw", "segmented", "heatmap")},
                "recommendation": r.get("recommendation", {}),
            })
            total_processed += 1

        except Exception as e:
            logger.error(f"Batch item error ({f.filename}): {e}")
            total_failed += 1
            results.append({"filename": f.filename, "error": str(e)})

    summary = {
        "total_processed": total_processed,
        "total_failed":    total_failed,
        "total_images":    len(files),
        "species_counts":  {},
        "avg_confidence":  0.0,
    }
    ok = [r for r in results if "error" not in r]
    if ok:
        for r in ok:
            sp = r["species"]
            summary["species_counts"][sp] = summary["species_counts"].get(sp, 0) + 1
        summary["avg_confidence"] = round(
            sum(r["confidence"] for r in ok) / len(ok), 3)

    return jsonify({"results": results, "summary": summary})


@app.route("/api/webcam_analyze", methods=["POST"])
def webcam_analyze():
    global _last_result
    try:
        data = request.get_json(force=True)
        b64  = data.get("image_b64", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        from utils.pipeline import run_pipeline
        result = run_pipeline(img)
        _last_result = result
        return jsonify(result)
    except Exception as e:
        logger.error(f"/api/webcam_analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/lime_explain", methods=["POST"])
def lime_explain():
    """LIME explanation — always returns 200."""
    try:
        img = _read_image_from_request("image")
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Get mask from pipeline
        mask = None
        try:
            from utils.pipeline import preprocess, segment
            img_proc, rgb, _ = preprocess(img)
            mask, _ = segment(rgb, False)
        except Exception:
            mask = np.ones((480, 640), dtype=np.uint8) * 255

        from utils.explainer import explain_prediction
        result = explain_prediction(img, mask)
        return jsonify(result)

    except Exception as e:
        logger.error(f"/api/lime_explain error: {e}", exc_info=True)
        from utils.explainer import _fallback_saliency
        try:
            fallback = _fallback_saliency(
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((480, 640), dtype=np.uint8)
            )
            return jsonify(fallback)
        except Exception:
            return jsonify({
                "original": "", "explanation": "", "method": "error"
            })


@app.route("/api/train", methods=["POST"])
def train():
    try:
        from utils.database import get_training_samples
        from utils.trainer import train_ensemble

        algo = request.json.get("algorithm", "both") if request.is_json else "both"
        samples = get_training_samples()

        if len(samples) < 10:
            return jsonify({"error": f"Only {len(samples)} samples in DB. Need at least 10."}), 400

        metrics = train_ensemble(samples, algorithm_choice=algo)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"/api/train error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/label_sample", methods=["POST"])
def label_sample():
    try:
        data = request.get_json(force=True)
        label = int(data.get("label", 0))
        features = {k: _safe_float(v) for k, v in data.items() if k != "label"}

        from utils.database import save_training_sample
        ok = save_training_sample(features, label)
        if ok:
            return jsonify({"saved": True})
        return jsonify({"saved": False, "error": "DB unavailable (offline mode)"}), 200
    except Exception as e:
        logger.error(f"/api/label_sample error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/seed_training", methods=["POST"])
def seed_training():
    try:
        from utils.database import seed_synthetic_samples, init_db
        init_db()
        n = seed_synthetic_samples()
        return jsonify({"inserted": n, "message": f"Seeded {n} synthetic training samples."})
    except Exception as e:
        logger.error(f"/api/seed_training error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/ablation", methods=["POST"])
def ablation():
    try:
        img = _read_image_from_request("image")
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        from utils.pipeline import preprocess, segment, morph_refine, extract_features
        from utils.segmentation import apply_morphology
        import time

        img_proc, rgb, _ = preprocess(img)

        configs = [
            {"se_shape": "ellipse", "se_size": 5,  "label": "Ellipse SE-5"},
            {"se_shape": "ellipse", "se_size": 11, "label": "Ellipse SE-11"},
            {"se_shape": "rect",    "se_size": 9,  "label": "Rect SE-9"},
            {"se_shape": "ellipse", "se_size": 9,  "label": "Ellipse SE-9 (ref)"},
        ]

        # ── PASS 1: collect all masks and metrics ────────────────────────────
        raw_results = []
        saved_masks = []

        for cfg in configs:
            t0 = time.time()
            base_mask, _ = segment(rgb, False)
            morph = apply_morphology(base_mask, cfg["se_shape"], cfg["se_size"])
            final_mask = morph["final"]
            feats = extract_features(final_mask, rgb, 0, 0)
            elapsed = round(time.time() - t0, 3)

            pix = int(final_mask.sum() / 255)
            total_pix = final_mask.shape[0] * final_mask.shape[1]

            raw_results.append({
                "label":       cfg["label"],
                "se_shape":    cfg["se_shape"],
                "se_size":     cfg["se_size"],
                "area":        round(feats.get("area", 0), 1),
                "circularity": round(feats.get("circularity", 0), 4),
                "solidity":    round(feats.get("solidity", 0), 4),
                "pct_leaf":    round(100.0 * pix / (total_pix + 1e-6), 2),
                "time_s":      elapsed,
            })
            saved_masks.append(final_mask)

        # ── PASS 2: compute IoU/Dice against reference (index 3) ─────────────
        ref_mask = saved_masks[3]
        final_results = []

        for i, (res, mask) in enumerate(zip(raw_results, saved_masks)):
            if i == 3:
                iou, dice = 1.0, 1.0
            else:
                iou, dice = _compute_iou_dice(mask, ref_mask)
            final_results.append({**res, "iou": round(iou, 4), "dice": round(dice, 4)})

        return jsonify({"configs": final_results})

    except Exception as e:
        logger.error(f"/api/ablation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _compute_iou_dice(mask_a: np.ndarray, mask_b: np.ndarray):
    a = (mask_a > 0).astype(bool).flatten()
    b = (mask_b > 0).astype(bool).flatten()
    intersection = (a & b).sum()
    union = (a | b).sum()
    iou  = intersection / (union + 1e-8)
    dice = 2 * intersection / (a.sum() + b.sum() + 1e-8)
    return float(iou), float(dice)


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    try:
        pred_file  = request.files.get("predicted")
        truth_file = request.files.get("ground_truth")
        if not pred_file or not truth_file:
            return jsonify({"error": "Need predicted and ground_truth mask images"}), 400

        def _read_mask(f):
            data = f.read()
            arr  = np.frombuffer(data, dtype=np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            return (img > 127).astype(np.uint8) if img is not None else None

        pred_mask  = _read_mask(pred_file)
        truth_mask = _read_mask(truth_file)

        if pred_mask is None or truth_mask is None:
            return jsonify({"error": "Cannot decode mask images"}), 400

        # Resize to same size
        if pred_mask.shape != truth_mask.shape:
            truth_mask = cv2.resize(truth_mask, (pred_mask.shape[1], pred_mask.shape[0]))

        iou, dice = _compute_iou_dice(pred_mask, truth_mask)

        if iou >= 0.8:
            rating = "Excellent"
        elif iou >= 0.6:
            rating = "Good"
        elif iou >= 0.4:
            rating = "Fair"
        else:
            rating = "Poor"

        return jsonify({"iou": round(iou, 4), "dice": round(dice, 4), "rating": rating})

    except Exception as e:
        logger.error(f"/api/evaluate error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/training_stats", methods=["GET"])
def training_stats():
    try:
        from utils.database import get_training_stats
        from utils.trainer import model_exists
        stats = get_training_stats()
        stats["model_ready"] = model_exists()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"total": 0, "by_label": {}, "model_ready": False})


@app.route("/api/history", methods=["GET"])
def history():
    try:
        from utils.database import get_history
        rows = get_history(200)
        return jsonify({"rows": rows})
    except Exception as e:
        return jsonify({"rows": []})


@app.route("/api/trend_data", methods=["GET"])
def trend_data():
    try:
        from utils.database import get_trend_data
        rows = get_trend_data()
        return jsonify({"data": rows})
    except Exception as e:
        return jsonify({"data": []})


@app.route("/api/generate_pdf", methods=["GET"])
def generate_pdf():
    global _last_result
    try:
        from utils.report import generate_pdf_report
        if not _last_result:
            return jsonify({"error": "No analysis result available"}), 400
        pdf_bytes = generate_pdf_report(_last_result)
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment; filename=leaflab_report.pdf"}
        )
    except Exception as e:
        logger.error(f"/api/generate_pdf error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/db_status", methods=["GET"])
def db_status():
    try:
        from utils.database import check_db_status
        ok = check_db_status()
        return jsonify({"connected": ok})
    except Exception:
        return jsonify({"connected": False})


if __name__ == "__main__":
    # Init DB on startup
    try:
        from utils.database import init_db
        init_db()
    except Exception as e:
        logger.warning(f"DB init skipped: {e}")
    app.run(debug=True, host="0.0.0.0", port=5000)
