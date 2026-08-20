"""Color anomaly detection — find small colored regions unusual for the scene.

Scene-relative approach: builds a color distribution from the image itself,
then flags small clusters of pixels whose color is statistically rare.
No predefined color targets — the terrain defines "normal".

This is the canonical implementation. color_anomalies.py is now a thin CLI
wrapper around this module.

Extracted from alpine_zoom.video (source of truth).
"""
import cv2
import numpy as np

# ── Parameters ─────────────────────────────────────────────────────────

MIN_COLORFULNESS = 10.0    # LAB distance from gray (128,128) to count as "colorful"
HIST_BINS = 32              # 2D histogram bins per channel (32x32 = 1024 bins)
RARITY_PERCENTILE = 97.0   # Flag top 3% rarest colorful pixels
MIN_AREA = 50               # Min cluster size in pixels (filters noise)
MIN_RARITY = 2.5            # Min average rarity for a finding (filters noise)
MAX_AREA_FRAC = 0.01        # Max cluster size as fraction of image (1%)
MORPH_OPEN_KERNEL = 3       # Opening kernel size (noise removal)
MORPH_CLOSE_KERNEL = 5      # Closing kernel size (fill gaps)
IGNORE_TOP = 100            # Ignore top N pixels (text markers)
IGNORE_BOTTOM = 30          # Ignore bottom N pixels
IGNORE_LEFT = 250           # Ignore left N pixels (text markers)


def zone_grid(h, w, rows=3, cols=3):
    """Build list of (zone_id, x0, y0, x1, y1) tuples for a 3x3 grid."""
    zones = []
    for r in range(rows):
        for c in range(cols):
            zid = r * cols + c + 1
            x0 = c * w // cols
            y0 = r * h // rows
            x1 = (c + 1) * w // cols if c < cols - 1 else w
            y1 = (r + 1) * h // rows if r < rows - 1 else h
            zones.append((zid, x0, y0, x1, y1))
    return zones


def build_ignore_mask(h, w):
    """Mask out text regions and grid lines."""
    mask = np.ones((h, w), dtype=np.uint8)
    mask[:IGNORE_TOP, :] = 0       # top text markers (filename, scene, time)
    mask[-IGNORE_BOTTOM:, :] = 0   # bottom text
    mask[:, :IGNORE_LEFT] = 0      # left text markers (GPS, telemetry)
    return mask


def estimate_color(a_val, b_val):
    """Estimate color name from LAB a,b channels (centered at 128).

    a = a_val - 128  (positive = red/magenta, negative = green)
    b = b_val - 128  (positive = yellow, negative = blue)

    This is the canonical implementation — matches alpine_zoom.video.
    The standalone color_anomalies.py had a different threshold ordering
    which produced different color names for the same LAB values. Fixed.
    """
    a = a_val - 128
    b = b_val - 128
    if a < -5 and b < 15:
        return "blue"
    if a > 20 and b > 10:
        return "orange"
    if a > 5 and b > 25:
        return "yellow"
    if a > 20 and b < 10:
        return "red"
    if a < -10 and b > 10:
        return "green"
    if a > 15 and b < -5:
        return "magenta"
    return "unknown"


def _bbox_iou(b1, b2):
    """Intersection-over-union of two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1)


def deduplicate(findings):
    """Merge findings with overlapping bounding boxes (IoU > 0.3)."""
    if len(findings) <= 1:
        return findings

    merged = []
    used = set()

    for i, f in enumerate(findings):
        if i in used:
            continue
        for j in range(i + 1, len(findings)):
            if j in used:
                continue
            if _bbox_iou(f["bbox"], findings[j]["bbox"]) > 0.3:
                # Merge j into i — keep higher confidence
                if findings[j]["confidence"] > f["confidence"]:
                    f = {**findings[j], "area": f["area"] + findings[j]["area"]}
                else:
                    f["area"] = f["area"] + findings[j]["area"]
                used.add(j)
        merged.append(f)
    return merged


def detect_color_anomalies(img, debug=False):
    """Detect small colored regions whose color is rare for this scene.

    Args:
        img: BGR OpenCV image
        debug: if True, returns (findings, anomaly_mask, colorfulness, rarity)

    Returns:
        findings: list of dicts with color, zone, area, bbox, centroid,
                  confidence, colorfulness, rarity, avg_lab
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    a_ch = lab[:, :, 1]  # green-red axis
    b_ch = lab[:, :, 2]  # blue-yellow axis

    # Colorfulness: distance from gray (128, 128)
    colorfulness = np.sqrt((a_ch - 128) ** 2 + (b_ch - 128) ** 2)

    # Only consider colorful pixels (not grey rock/snow)
    colorful_mask = colorfulness > MIN_COLORFULNESS
    n_colorful = np.count_nonzero(colorful_mask)

    zones = zone_grid(h, w)
    ignore = build_ignore_mask(h, w)

    if n_colorful < 50:
        # Very few colorful pixels — all of them are anomalous
        rarity = np.zeros_like(a_ch)
        anomaly_mask = (colorful_mask.astype(np.uint8) * ignore)
    else:
        # Build 2D chromaticity histogram from colorful pixels only
        a_idx = np.clip((a_ch / 256 * HIST_BINS).astype(np.int32), 0, HIST_BINS - 1)
        b_idx = np.clip((b_ch / 256 * HIST_BINS).astype(np.int32), 0, HIST_BINS - 1)

        hist = np.zeros((HIST_BINS, HIST_BINS), dtype=np.float32)
        np.add.at(hist, (a_idx[colorful_mask], b_idx[colorful_mask]), 1)

        # Smooth histogram to reduce noise
        hist = cv2.GaussianBlur(hist, (3, 3), 0)

        # Normalize to probability density
        total = hist.sum()
        hist_norm = hist / max(total, 1)

        # Compute rarity for each pixel: -log(density)
        density = hist_norm[a_idx, b_idx]
        rarity = -np.log(density + 1e-10)

        # Only keep colorful pixels
        rarity = rarity * colorful_mask

        # Adaptive threshold: top X% rarest
        rarity_vals = rarity[colorful_mask]
        if len(rarity_vals) == 0:
            if debug:
                return [], None, colorfulness, None
            return []
        thresh = np.percentile(rarity_vals, RARITY_PERCENTILE)
        anomaly_mask = (rarity > thresh).astype(np.uint8) * ignore

    # Morphological cleanup
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN,
                                     np.ones((MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL), np.uint8))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE,
                                     np.ones((MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL), np.uint8))

    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(anomaly_mask, connectivity=8)

    max_area = h * w * MAX_AREA_FRAC
    findings = []

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA or area > max_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        cluster_mask = labels == i
        # Filter by rarity (noise has low rarity)
        cluster_rarity = float(np.mean(rarity[cluster_mask])) if n_colorful >= 50 else 0.0
        if cluster_rarity < MIN_RARITY:
            continue
        avg_a = float(np.mean(a_ch[cluster_mask]))
        avg_b = float(np.mean(b_ch[cluster_mask]))
        avg_colorfulness = float(np.mean(colorfulness[cluster_mask]))
        avg_rarity = float(np.mean(rarity[cluster_mask])) if n_colorful >= 50 else 0.0

        # Determine zone
        zone_id = 5
        for zid, x0, y0, x1, y1 in zones:
            if x0 <= cx < x1 and y0 <= cy < y1:
                zone_id = zid
                break

        color_name = estimate_color(avg_a, avg_b)

        # Confidence: based on colorfulness, rarity, and compactness
        compactness = area / max(bw * bh, 1)  # filled fraction of bbox
        conf = min(1.0, avg_colorfulness / 80 * 0.3 +
                   min(avg_rarity / 5, 1.0) * 0.4 +
                   compactness * 0.3)

        findings.append({
            "color": color_name,
            "zone": zone_id,
            "area": int(area),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "centroid": (int(cx), int(cy)),
            "confidence": round(conf, 2),
            "colorfulness": round(avg_colorfulness, 1),
            "rarity": round(avg_rarity, 2),
            "avg_lab": (round(avg_a, 1), round(avg_b, 1)),
        })

    # Deduplicate: merge overlapping detections (IoU > 0.3)
    findings = deduplicate(findings)

    # Sort by confidence descending
    findings.sort(key=lambda f: -f["confidence"])

    if debug:
        return findings, anomaly_mask, colorfulness, rarity if n_colorful >= 50 else None
    return findings


def draw_color_findings(img, findings):
    """Draw bounding boxes and labels for color anomalies."""
    result = img.copy()
    color_map = {
        "orange": (0, 165, 255), "yellow": (0, 255, 255), "red": (0, 0, 255),
        "blue": (255, 0, 0), "green": (0, 255, 0), "magenta": (255, 0, 255),
        "unknown": (200, 200, 200),
    }
    for f in findings:
        x, y, bw, bh = f["bbox"]
        box_color = color_map.get(f["color"], (0, 255, 0))
        cv2.rectangle(result, (x - 2, y - 2), (x + bw + 2, y + bh + 2), box_color, 2)
        label = f"{f['color']} Z{f['zone']} conf={f['confidence']:.1f} A={f['area']}"
        cv2.putText(result, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
    return result
