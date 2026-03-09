"""
LeafLab — Ensemble ML Trainer
k-NN + Random Forest hard-voting ensemble with StandardScaler.
"""
import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP = {
    0: "Healthy Broad Leaf",
    1: "Elongated Narrow Leaf",
    2: "Lobed/Serrated Leaf",
}

FEATURE_ORDER = [
    "area", "perimeter", "circularity", "aspect_ratio", "solidity",
    "extent", "eccentricity", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5",
    "hu_6", "hu_7", "glcm_contrast", "glcm_energy", "glcm_homogeneity",
    "glcm_correlation", "mean_exg", "mean_hue", "skeleton_length",
    "branch_points",
]

MODEL_PATH  = "models/ensemble_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def _features_to_array(features: dict) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float64)


def train_ensemble(samples: list, algorithm_choice: str = "both") -> dict:
    """
    Train k-NN + RF ensemble.
    samples: list of dicts with feature keys + 'label' key.
    Returns metrics dict.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, confusion_matrix

    if len(samples) < 10:
        return {"error": f"Need at least 10 samples, got {len(samples)}"}

    X = np.array([_features_to_array(s) for s in samples])
    y = np.array([int(s["label"]) for s in samples])

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return {"error": "Need samples from at least 2 classes"}

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build classifiers based on choice
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    if algorithm_choice == "knn":
        model = knn
        model_name = "k-NN (k=5)"
    elif algorithm_choice == "rf":
        model = rf
        model_name = "Random Forest (100 trees)"
    else:
        model = VotingClassifier(
            estimators=[("knn", knn), ("rf", rf)],
            voting="hard"
        )
        model_name = "Ensemble (k-NN + RF)"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm  = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

    # 5-fold CV on full data (may fail if too few samples per class)
    try:
        cv_scores = cross_val_score(model, X_scaled, y,
                                    cv=StratifiedKFold(n_splits=min(5, len(unique) * min(counts))),
                                    scoring="accuracy")
        cv_acc = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
    except Exception as e:
        logger.warning(f"CV failed: {e}")
        cv_acc = acc
        cv_std = 0.0

    # Per-class accuracy
    per_class = {}
    for label_id in [0, 1, 2]:
        mask = y_test == label_id
        if mask.sum() > 0:
            per_class[LABEL_MAP[label_id]] = float((y_pred[mask] == label_id).mean())
        else:
            per_class[LABEL_MAP[label_id]] = None

    # Save
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return {
        "model_name":    model_name,
        "accuracy":      round(acc, 4),
        "cv_accuracy":   round(cv_acc, 4),
        "cv_std":        round(cv_std, 4),
        "confusion_matrix": cm,
        "per_class":     per_class,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "n_total":       len(X),
        "label_counts":  dict(zip([int(u) for u in unique], [int(c) for c in counts])),
    }


def predict_leaf(features: dict, model_path: str = MODEL_PATH,
                 scaler_path: str = SCALER_PATH):
    """Load model + scaler and predict. Returns (species_name, confidence)."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    x = _features_to_array(features).reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_scaled = scaler.transform(x)

    pred = int(model.predict(x_scaled)[0])
    species = LABEL_MAP.get(pred, "Unknown")

    # Confidence: use predict_proba if available
    conf = 0.75
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_scaled)[0]
            conf = float(proba.max())
        except Exception:
            pass
    elif hasattr(model, "estimators_"):
        # Voting: tally votes
        try:
            votes = np.array([est.predict(x_scaled)[0] for est in model.estimators_])
            conf = float((votes == pred).mean())
        except Exception:
            pass

    return species, round(conf, 3)


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
