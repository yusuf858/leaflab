"""
LeafLab — Database Layer
MySQL via pymysql with graceful offline fallback.
"""
import os
import json
import time
import logging

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", ""),
    "database": os.getenv("DB_NAME", "leaflab_db"),
    "charset":  "utf8mb4",
    "autocommit": False,
}

_db_available = None


def _get_connection():
    try:
        import pymysql
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.debug(f"DB connection failed: {e}")
        return None


def check_db_status() -> bool:
    """Return True if DB is reachable."""
    global _db_available
    conn = _get_connection()
    if conn:
        conn.close()
        _db_available = True
        return True
    _db_available = False
    return False


def init_db() -> bool:
    """Create tables if they don't exist."""
    conn = _get_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analysis_log (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    filename        VARCHAR(255),
                    species         VARCHAR(128),
                    confidence      FLOAT,
                    area            FLOAT,
                    perimeter       FLOAT,
                    circularity     FLOAT,
                    solidity        FLOAT,
                    aspect_ratio    FLOAT,
                    process_time    FLOAT,
                    location        VARCHAR(255),
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shape_features_log (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    analysis_id     INT,
                    feature_name    VARCHAR(64),
                    feature_value   DOUBLE,
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_samples (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    label           INT NOT NULL,
                    area            DOUBLE DEFAULT 0,
                    perimeter       DOUBLE DEFAULT 0,
                    circularity     DOUBLE DEFAULT 0,
                    aspect_ratio    DOUBLE DEFAULT 0,
                    solidity        DOUBLE DEFAULT 0,
                    extent          DOUBLE DEFAULT 0,
                    eccentricity    DOUBLE DEFAULT 0,
                    hu_1            DOUBLE DEFAULT 0,
                    hu_2            DOUBLE DEFAULT 0,
                    hu_3            DOUBLE DEFAULT 0,
                    hu_4            DOUBLE DEFAULT 0,
                    hu_5            DOUBLE DEFAULT 0,
                    hu_6            DOUBLE DEFAULT 0,
                    hu_7            DOUBLE DEFAULT 0,
                    glcm_contrast   DOUBLE DEFAULT 0,
                    glcm_energy     DOUBLE DEFAULT 0,
                    glcm_homogeneity DOUBLE DEFAULT 0,
                    glcm_correlation DOUBLE DEFAULT 0,
                    mean_exg        DOUBLE DEFAULT 0,
                    mean_hue        DOUBLE DEFAULT 0,
                    skeleton_length DOUBLE DEFAULT 0,
                    branch_points   DOUBLE DEFAULT 0,
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            """)
        conn.close()
        return True
    except Exception as e:
        logger.error(f"init_db error: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return False


def log_analysis(filename, species, confidence, features: dict, location="") -> int:
    """Insert into analysis_log. Returns inserted id or -1."""
    conn = _get_connection()
    if not conn:
        return -1
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analysis_log
                    (filename, species, confidence, area, perimeter,
                     circularity, solidity, aspect_ratio, process_time, location)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                filename, species, confidence,
                features.get("area", 0),
                features.get("perimeter", 0),
                features.get("circularity", 0),
                features.get("solidity", 0),
                features.get("aspect_ratio", 0),
                features.get("process_time", 0),
                location
            ))
            analysis_id = cur.lastrowid
            # log all features
            for k, v in features.items():
                try:
                    cur.execute("""
                        INSERT INTO shape_features_log (analysis_id, feature_name, feature_value)
                        VALUES (%s,%s,%s)
                    """, (analysis_id, k, float(v)))
                except Exception:
                    pass
        conn.commit()
        conn.close()
        return analysis_id
    except Exception as e:
        logger.error(f"log_analysis error: {e}")
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        return -1


def get_history(limit=100) -> list:
    """Return list of analysis_log rows."""
    conn = _get_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, filename, species, confidence,
                       area, perimeter, circularity, solidity,
                       aspect_ratio, process_time, location,
                       DATE_FORMAT(created_at,'%%Y-%%m-%%d %%H:%%i') as ts
                FROM analysis_log
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"get_history error: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return []


def get_trend_data() -> list:
    """Return date + avg_confidence for charting."""
    conn = _get_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DATE(created_at) as d, ROUND(AVG(confidence),3) as avg_conf,
                       COUNT(*) as cnt
                FROM analysis_log
                GROUP BY DATE(created_at)
                ORDER BY d DESC
                LIMIT 30
            """)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"get_trend_data error: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return []


def get_training_samples() -> list:
    """Return all rows from training_samples."""
    conn = _get_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM training_samples ORDER BY id DESC")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"get_training_samples error: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return []


def get_training_stats() -> dict:
    """Return counts per label."""
    conn = _get_connection()
    if not conn:
        return {"total": 0, "by_label": {}}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT label, COUNT(*) as cnt
                FROM training_samples
                GROUP BY label
            """)
            rows = cur.fetchall()
        conn.close()
        by_label = {str(r[0]): r[1] for r in rows}
        total = sum(v for v in by_label.values())
        return {"total": total, "by_label": by_label}
    except Exception as e:
        logger.error(f"get_training_stats error: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return {"total": 0, "by_label": {}}


def save_training_sample(features: dict, label: int) -> bool:
    """Insert one row into training_samples."""
    conn = _get_connection()
    if not conn:
        return False
    cols = [
        "label","area","perimeter","circularity","aspect_ratio","solidity",
        "extent","eccentricity","hu_1","hu_2","hu_3","hu_4","hu_5","hu_6","hu_7",
        "glcm_contrast","glcm_energy","glcm_homogeneity","glcm_correlation",
        "mean_exg","mean_hue","skeleton_length","branch_points"
    ]
    vals = [label] + [float(features.get(c, 0)) for c in cols[1:]]
    placeholders = ",".join(["%s"] * len(cols))
    col_names = ",".join(cols)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO training_samples ({col_names}) VALUES ({placeholders})", vals)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"save_training_sample error: {e}")
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        return False


def seed_synthetic_samples() -> int:
    """Insert 120 synthetic training samples. Return count inserted."""
    import random
    random.seed(42)

    # Always try to init DB first
    init_db()

    conn = _get_connection()
    if not conn:
        return 0

    def rand(lo, hi):
        return round(random.uniform(lo, hi), 4)

    samples = []
    # label 0: Healthy Broad Leaf
    for _ in range(40):
        area = rand(8000, 45000)
        samples.append({
            "label": 0, "area": area,
            "perimeter": rand(300, 900),
            "circularity": rand(0.65, 0.92),
            "aspect_ratio": rand(1.0, 1.8),
            "solidity": rand(0.85, 0.98),
            "extent": rand(0.6, 0.82),
            "eccentricity": rand(0.2, 0.6),
            "hu_1": rand(0.1, 0.3), "hu_2": rand(0.0, 0.05),
            "hu_3": rand(0.0, 0.01), "hu_4": rand(0.0, 0.005),
            "hu_5": rand(0.0, 0.001), "hu_6": rand(0.0, 0.002),
            "hu_7": rand(0.0, 0.0005),
            "glcm_contrast": rand(0.1, 2.0), "glcm_energy": rand(0.1, 0.5),
            "glcm_homogeneity": rand(0.5, 0.9), "glcm_correlation": rand(0.7, 0.99),
            "mean_exg": rand(10, 60), "mean_hue": rand(50, 90),
            "skeleton_length": rand(500, 2000), "branch_points": rand(5, 30),
        })
    # label 1: Elongated Narrow Leaf
    for _ in range(40):
        area = rand(5000, 30000)
        samples.append({
            "label": 1, "area": area,
            "perimeter": rand(400, 1200),
            "circularity": rand(0.15, 0.42),
            "aspect_ratio": rand(2.5, 6.0),
            "solidity": rand(0.78, 0.92),
            "extent": rand(0.4, 0.65),
            "eccentricity": rand(0.7, 0.98),
            "hu_1": rand(0.2, 0.5), "hu_2": rand(0.01, 0.1),
            "hu_3": rand(0.0, 0.02), "hu_4": rand(0.0, 0.01),
            "hu_5": rand(0.0, 0.002), "hu_6": rand(0.0, 0.003),
            "hu_7": rand(0.0, 0.001),
            "glcm_contrast": rand(0.5, 3.0), "glcm_energy": rand(0.05, 0.3),
            "glcm_homogeneity": rand(0.3, 0.7), "glcm_correlation": rand(0.5, 0.9),
            "mean_exg": rand(5, 40), "mean_hue": rand(70, 120),
            "skeleton_length": rand(800, 3500), "branch_points": rand(2, 15),
        })
    # label 2: Lobed/Serrated Leaf
    for _ in range(40):
        area = rand(10000, 60000)
        samples.append({
            "label": 2, "area": area,
            "perimeter": rand(600, 2000),
            "circularity": rand(0.20, 0.55),
            "aspect_ratio": rand(1.2, 2.8),
            "solidity": rand(0.55, 0.78),
            "extent": rand(0.35, 0.6),
            "eccentricity": rand(0.4, 0.85),
            "hu_1": rand(0.15, 0.4), "hu_2": rand(0.005, 0.08),
            "hu_3": rand(0.0, 0.015), "hu_4": rand(0.0, 0.008),
            "hu_5": rand(0.0, 0.0015), "hu_6": rand(0.0, 0.0025),
            "hu_7": rand(0.0, 0.0008),
            "glcm_contrast": rand(1.0, 5.0), "glcm_energy": rand(0.03, 0.2),
            "glcm_homogeneity": rand(0.2, 0.6), "glcm_correlation": rand(0.4, 0.85),
            "mean_exg": rand(8, 50), "mean_hue": rand(40, 110),
            "skeleton_length": rand(1000, 5000), "branch_points": rand(15, 60),
        })

    inserted = 0
    try:
        # Use executemany for a single round-trip instead of 120 individual calls
        # This prevents proxy timeouts on shared/remote DB hosts (e.g. Filess.io)
        cols = list(samples[0].keys())
        col_names = ",".join(cols)
        placeholders = ",".join(["%s"] * len(cols))
        all_vals = [list(s.values()) for s in samples]

        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO training_samples ({col_names}) VALUES ({placeholders})",
                all_vals
            )
            inserted = len(all_vals)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"seed_synthetic_samples error: {e}")
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
    return inserted