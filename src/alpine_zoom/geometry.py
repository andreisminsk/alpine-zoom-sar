"""Geometry anomaly detection — find man-made geometric shapes in natural scenes.

Deterministic approach: uses edge detection, contour analysis, Hough transforms,
and polygon approximation to identify shapes that are too regular for natural
terrain (straight lines, right angles, rectangles, circles, regular polygons).

Usage:
  python geometry_anomalies.py <image_path> [--debug]
  python geometry_anomalies.py <scenes_dir> [--debug]

Example:
  python geometry_anomalies.py analysis_results/.../scene_352_f01890_orig.jpg --debug
"""
import sys
import os
import cv2
import numpy as np
import argparse
import math

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.common import canonical_base, collect_orig_images

# ── Parameters ─────────────────────────────────────────────────────────

CANNY_LOW = 50               # Canny lower threshold
CANNY_HIGH = 150             # Canny upper threshold
BLUR_KERNEL = 5              # Gaussian blur kernel size
MIN_CONTOUR_AREA = 80         # Min contour area in pixels
MAX_AREA_FRAC = 0.02          # Max shape size as fraction of image (2%)
MIN_LINE_LENGTH = 40         # Min line segment length (HoughLinesP)
MAX_LINE_GAP = 8            # Max gap for Hough line segments
HOUGH_THRESHOLD = 50         # Hough line accumulator threshold
HOUGH_CIRCLE_MIN_R = 15        # Min circle radius
HOUGH_CIRCLE_MAX_R = 80       # Max circle radius
HOUGH_CIRCLE_THRESH = 80     # Hough circle accumulator threshold
HOUGH_CIRCLE_MIN_DIST = 60   # Min distance between Hough circle centers
CIRCLE_GRAD_MEAN_MAX = 25.0  # Max mean gradient-radial angle diff (degrees)
CIRCLE_GRAD_STD_MAX = 18.0   # Max std of gradient-radial angle diff
CIRCLE_EDGE_THRESH = 60     # Edge intensity threshold for circle verification
CIRCLE_MIN_COVERAGE = 0.50   # Min fraction of circumference with edges
POLY_EPS_FRAC = 0.02          # approxPolyDP epsilon as fraction of perimeter
RIGHT_ANGLE_TOL = 12.0        # Degrees tolerance for right angle detection
PARALLEL_ANGLE_TOL = 10.0    # Degrees tolerance for parallel line detection
MIN_CIRCULARITY = 0.88        # Min circularity to classify as circle
MIN_RECT_SCORE = 0.85         # Min rectangularity score (0-1)
MIN_REGULARITY = 0.75         # Min polygon regularity (equal sides/angles)
MIN_CONFIDENCE = 0.55         # Min confidence to report a finding
MORPH_CLOSE_KERNEL = 3        # Closing kernel for edge cleanup

# Ignore regions (same as color_anomalies.py — text/telemetry overlays)
IGNORE_TOP = 100
IGNORE_BOTTOM = 30
IGNORE_LEFT = 250


def zone_grid(h, w, rows=3, cols=3):
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


def ignore_mask(h, w):
    """Mask out text regions and grid lines."""
    mask = np.ones((h, w), dtype=np.uint8)
    mask[:IGNORE_TOP, :] = 0
    mask[-IGNORE_BOTTOM:, :] = 0
    mask[:, :IGNORE_LEFT] = 0
    return mask


def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def line_angle(line):
    """Angle of a line segment (x1,y1,x2,y2) in degrees (0-180)."""
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def line_length(line):
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def compute_circularity(area, perimeter):
    """Circularity: 4*pi*area / perimeter^2. Circle = 1.0."""
    if perimeter < 1:
        return 0.0
    return 4 * math.pi * area / (perimeter * perimeter)


def compute_rect_score(contour):
    """Score how well a contour fits a rectangle (0-1).

    Based on: area ratio (contour area vs min area bounding rect),
    and aspect ratio consistency of opposite sides.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    rect_area = cv2.contourArea(box)
    contour_area = cv2.contourArea(contour)
    if rect_area < 1:
        return 0.0

    # Area fill ratio: how much of the bounding rect the contour fills
    fill_ratio = min(contour_area, rect_area) / max(contour_area, rect_area)

    # Side length consistency: opposite sides should be equal
    sides = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        sides.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    # sides[0] vs sides[2], sides[1] vs sides[3]
    opp_ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2] + 1e-10)
    opp_ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3] + 1e-10)
    side_consistency = (opp_ratio1 + opp_ratio2) / 2

    return fill_ratio * 0.5 + side_consistency * 0.5


def compute_polygon_regularity(vertices):
    """Score how regular a polygon is (equal sides + equal angles). 0-1."""
    n = len(vertices)
    if n < 3:
        return 0.0

    sides = []
    angles = []
    for i in range(n):
        p1 = np.array(vertices[i], dtype=np.float64)
        p2 = np.array(vertices[(i + 1) % n], dtype=np.float64)
        p0 = np.array(vertices[(i - 1) % n], dtype=np.float64)
        sides.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        v1 = p1 - p0
        v2 = p2 - p1
        angles.append(angle_between(v1, v2))

    sides = np.array(sides)
    angles = np.array(angles)
    mean_side = sides.mean()
    mean_angle = angles.mean()

    if mean_side < 1:
        return 0.0

    side_cv = np.std(sides) / mean_side  # coefficient of variation
    angle_cv = np.std(angles) / (mean_angle + 1e-10)

    # Lower variation = higher regularity
    regularity = 1.0 / (1.0 + side_cv * 2 + angle_cv * 2)
    return min(1.0, regularity)


def count_right_angles(vertices):
    """Count vertices that form approximately right angles."""
    n = len(vertices)
    if n < 3:
        return 0
    count = 0
    for i in range(n):
        p0 = np.array(vertices[(i - 1) % n], dtype=np.float64)
        p1 = np.array(vertices[i], dtype=np.float64)
        p2 = np.array(vertices[(i + 1) % n], dtype=np.float64)
        v1 = p1 - p0
        v2 = p2 - p1
        ang = angle_between(v1, v2)
        if abs(ang - 90) < RIGHT_ANGLE_TOL:
            count += 1
    return count


def classify_shape(contour):
    """Classify a contour into a geometric shape.

    Returns: (shape_name, score, vertices, extra_info)
    """
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return None, 0, None, {}

    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return None, 0, None, {}

    # Polygon approximation
    eps = POLY_EPS_FRAC * perimeter
    approx = cv2.approxPolyDP(contour, eps, True)
    vertices = [tuple(p[0]) for p in approx]
    n_verts = len(vertices)

    # Circularity check
    circularity = compute_circularity(area, perimeter)

    # Circle: high circularity, enough vertices in approximation
    if circularity > MIN_CIRCULARITY and n_verts >= 6:
        score = min(1.0, circularity)
        return "circle", score, vertices, {"circularity": round(circularity, 3)}

    # Line: 2 vertices, high aspect ratio
    if n_verts == 2:
        length = line_length(vertices[0] + vertices[1])
        score = min(1.0, length / 100)
        return "line", score, vertices, {"length": round(length, 1)}

    # Triangle: 3 vertices
    if n_verts == 3:
        regularity = compute_polygon_regularity(vertices)
        right_angles = count_right_angles(vertices)
        score = regularity * 0.7 + (right_angles / 3.0) * 0.3
        return "triangle", score, vertices, {
            "regularity": round(regularity, 3),
            "right_angles": right_angles,
        }

    # Rectangle / square: 4 vertices
    if n_verts == 4:
        rect_score = compute_rect_score(contour)
        right_angles = count_right_angles(vertices)
        regularity = compute_polygon_regularity(vertices)

        # Check if square (equal sides)
        sides = []
        for i in range(4):
            p1 = np.array(vertices[i], dtype=np.float64)
            p2 = np.array(vertices[(i + 1) % 4], dtype=np.float64)
            sides.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        side_ratio = min(sides) / max(sides + [1e-10])
        is_square = side_ratio > 0.85

        if rect_score > MIN_RECT_SCORE:
            shape = "square" if is_square else "rectangle"
            score = rect_score * 0.5 + (right_angles / 4.0) * 0.3 + regularity * 0.2
            return shape, score, vertices, {
                "rect_score": round(rect_score, 3),
                "right_angles": right_angles,
                "regularity": round(regularity, 3),
                "side_ratio": round(side_ratio, 3),
            }

    # Regular polygon: 5+ vertices
    if n_verts >= 5:
        regularity = compute_polygon_regularity(vertices)
        if regularity > MIN_REGULARITY:
            shape = f"polygon-{n_verts}"
            score = regularity
            return shape, score, vertices, {"regularity": round(regularity, 3)}

    return None, 0, None, {}


def detect_hough_lines(edges, ignore):
    """Detect line segments using HoughLinesP."""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            HOUGH_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)
    findings = []
    if lines is None:
        return findings

    for line in lines:
        x1, y1, x2, y2 = line[0] if line.ndim > 1 else line
        # Check if midpoint is in valid region
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if my >= ignore.shape[0] or mx >= ignore.shape[1]:
            continue
        if ignore[my, mx] == 0:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        if length < MIN_LINE_LENGTH:
            continue
        findings.append({
            "shape": "line",
            "zone": 0,  # assigned later
            "area": int(length),
            "bbox": (int(min(x1, x2)), int(min(y1, y2)),
                     int(abs(x2 - x1)), int(abs(y2 - y1))),
            "centroid": ((x1 + x2) // 2, (y1 + y2) // 2),
            "confidence": round(min(1.0, length / 100), 2),
            "length": round(length, 1),
            "angle": round(line_angle((x1, y1, x2, y2)), 1),
            "endpoints": ((int(x1), int(y1)), (int(x2), int(y2))),
        })
    return findings


def verify_circle_gradient(gray, cx, cy, r):
    """Check if edges around (cx, cy, r) have radially-consistent gradients.

    For a true circle, edge gradients point radially (toward/away from center).
    Natural features have chaotic gradient directions. Returns:
    (coverage, mean_diff, std_diff) — coverage is fraction of circumference
    with edges, mean_diff/std_diff measure gradient-radial angle consistency.
    """
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    n_samples = max(36, int(r * 2))
    angles = np.linspace(0, 2 * math.pi, n_samples, endpoint=False)
    band = max(2, int(r * 0.1))

    covered = 0
    radial_diffs = []

    for ang in angles:
        px = int(cx + r * math.cos(ang))
        py = int(cy + r * math.sin(ang))
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
        x0 = max(0, px - band)
        x1 = min(w, px + band + 1)
        y0 = max(0, py - band)
        y1 = min(h, py + band + 1)
        patch = edges[y0:y1, x0:x1]
        if patch.size > 0 and np.max(patch) > CIRCLE_EDGE_THRESH:
            covered += 1
            # Gradient direction at this point
            gxv = gx[py, px]
            gyv = gy[py, px]
            grad_ang = math.atan2(gyv, gxv)
            # Expected radial direction (center to point)
            radial_ang = ang
            # Angle between gradient and radial (parallel = 0)
            diff = abs((grad_ang - radial_ang + math.pi) % (2 * math.pi) - math.pi)
            diff = min(diff, math.pi - diff)
            radial_diffs.append(math.degrees(diff))

    coverage = covered / n_samples
    if len(radial_diffs) < 10:
        return coverage, 90.0, 90.0

    mean_diff = float(np.mean(radial_diffs))
    std_diff = float(np.std(radial_diffs))
    return coverage, mean_diff, std_diff


def detect_hough_circles(gray, ignore):
    """Detect circles using HoughCircles with gradient direction verification."""
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, HOUGH_CIRCLE_MIN_DIST,
                               param1=CANNY_HIGH, param2=HOUGH_CIRCLE_THRESH,
                               minRadius=HOUGH_CIRCLE_MIN_R,
                               maxRadius=HOUGH_CIRCLE_MAX_R)
    findings = []
    if circles is None:
        return findings

    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if cy >= ignore.shape[0] or cx >= ignore.shape[1]:
            continue
        if ignore[cy, cx] == 0:
            continue

        # Verify gradient direction consistency
        coverage, mean_diff, std_diff = verify_circle_gradient(gray, cx, cy, r)
        if coverage < CIRCLE_MIN_COVERAGE:
            continue
        if mean_diff > CIRCLE_GRAD_MEAN_MAX or std_diff > CIRCLE_GRAD_STD_MAX:
            continue

        area = int(math.pi * r * r)
        # Confidence scales with radius and gradient consistency
        conf = min(1.0, max(0.0, (r - HOUGH_CIRCLE_MIN_R) / 30 + 0.4))
        grad_score = 1.0 - (mean_diff / 90.0)  # 0=perfect, 90=chaotic
        conf = conf * (0.5 + 0.5 * grad_score)
        findings.append({
            "shape": "circle",
            "zone": 0,
            "area": area,
            "bbox": (cx - r, cy - r, 2 * r, 2 * r),
            "centroid": (cx, cy),
            "confidence": round(conf, 2),
            "radius": r,
            "circularity": 1.0,
            "edge_coverage": round(coverage, 3),
            "grad_mean_diff": round(mean_diff, 1),
            "grad_std_diff": round(std_diff, 1),
        })
    return findings


def detect_geometry_anomalies(img, debug=False):
    """Detect man-made geometric shapes in the image.

    Returns list of findings: [{shape, zone, area, bbox, confidence, ...}]
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                   np.ones((MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL), np.uint8))

    ignore = ignore_mask(h, w)
    zones = zone_grid(h, w)

    findings = []

    # ── Contour-based shape detection ──────────────────────────────
    contours, _ = cv2.findContours(edges_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = h * w * MAX_AREA_FRAC
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > max_area:
            continue

        shape_name, score, vertices, extra = classify_shape(contour)
        if shape_name is None or score < 0.3:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = x + bw // 2, y + bh // 2

        # Check ignore mask
        if cy >= h or cx >= w or ignore[cy, cx] == 0:
            continue

        # Determine zone
        zone_id = 5
        for zid, x0, y0, x1, y1 in zones:
            if x0 <= cx < x1 and y0 <= cy < y1:
                zone_id = zid
                break

        # Confidence: shape score + size significance
        size_factor = min(1.0, area / 500)
        confidence = min(1.0, score * 0.7 + size_factor * 0.3)

        if confidence < MIN_CONFIDENCE:
            continue

        findings.append({
            "shape": shape_name,
            "zone": zone_id,
            "area": int(area),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "centroid": (int(cx), int(cy)),
            "confidence": round(confidence, 2),
            "score": round(score, 3),
            **{k: v for k, v in extra.items()},
        })

    # ── Hough line detection ───────────────────────────────────────
    hough_lines = detect_hough_lines(edges_clean, ignore)
    for f in hough_lines:
        cx, cy = f["centroid"]
        zone_id = 5
        for zid, x0, y0, x1, y1 in zones:
            if x0 <= cx < x1 and y0 <= cy < y1:
                zone_id = zid
                break
        f["zone"] = zone_id
        if f["confidence"] >= MIN_CONFIDENCE:
            findings.append(f)

    # ── Hough circle detection ────────────────────────────────────
    hough_circles = detect_hough_circles(gray, ignore)
    for f in hough_circles:
        cx, cy = f["centroid"]
        zone_id = 5
        for zid, x0, y0, x1, y1 in zones:
            if x0 <= cx < x1 and y0 <= cy < y1:
                zone_id = zid
                break
        f["zone"] = zone_id
        if f["confidence"] >= MIN_CONFIDENCE:
            findings.append(f)

    # Deduplicate
    findings = deduplicate(findings)

    # Sort by confidence descending
    findings.sort(key=lambda f: -f["confidence"])

    if debug:
        return findings, edges_clean, contours
    return findings


def line_similarity(f1, f2):
    """Check if two line findings are near-parallel and co-located.

    Returns True if lines should be merged (same angle, close position).
    """
    if "endpoints" not in f1 or "endpoints" not in f2:
        return False

    ang1 = f1.get("angle", 0)
    ang2 = f2.get("angle", 0)
    # Angular difference (0-180, account for wraparound)
    ang_diff = abs(ang1 - ang2)
    ang_diff = min(ang_diff, 180 - ang_diff)

    if ang_diff > PARALLEL_ANGLE_TOL:
        return False

    # Perpendicular distance between line midpoints projected onto the
    # normal of the first line
    cx1, cy1 = f1["centroid"]
    cx2, cy2 = f2["centroid"]
    # Normal direction (perpendicular to line angle)
    rad = math.radians(ang1)
    nx, ny = -math.sin(rad), math.cos(rad)
    perp_dist = abs((cx2 - cx1) * nx + (cy2 - cy1) * ny)

    # Also check overlap along the line direction
    dx, dy = math.cos(rad), math.sin(rad)
    proj1 = (cx1) * dx + (cy1) * dy
    proj2 = (cx2) * dx + (cy2) * dy
    len1 = f1.get("length", 0)
    len2 = f2.get("length", 0)
    overlap = min(proj1 + len1 / 2, proj2 + len2 / 2) - max(proj1 - len1 / 2, proj2 - len2 / 2)

    # Merge if perpendicular distance is small and lines overlap
    max_perp_dist = max(8, min(len1, len2) * 0.15)
    return perp_dist < max_perp_dist and overlap > 0


def deduplicate(findings):
    """Merge findings with overlapping bounding boxes (IoU > 0.3) or similar lines."""
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
            # Only merge same-shape findings
            if f.get("shape") != findings[j].get("shape"):
                continue

            # Line-specific similarity check (IoU fails on degenerate bboxes)
            if f.get("shape") == "line":
                similar = line_similarity(f, findings[j])
            else:
                similar = bbox_iou(f["bbox"], findings[j]["bbox"]) > 0.3

            if similar:
                if f["confidence"] >= findings[j]["confidence"]:
                    used.add(j)
                else:
                    used.add(i)
                    break
        if i not in used:
            merged.append(f)

    return merged


def bbox_iou(b1, b2):
    """Compute IoU of two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1)


def draw_findings(img, findings):
    """Draw bounding boxes and labels for detected shapes."""
    result = img.copy()
    shape_colors = {
        "line": (0, 255, 255),      # yellow
        "rectangle": (0, 165, 255),  # orange
        "square": (0, 200, 255),     # light orange
        "circle": (255, 0, 0),       # blue
        "triangle": (0, 255, 0),     # green
        "magenta": (255, 0, 255),
    }

    for f in findings:
        x, y, bw, bh = f["bbox"]
        shape = f["shape"]
        color = shape_colors.get(shape, (200, 200, 200))

        if shape == "line" and "endpoints" in f:
            p1, p2 = f["endpoints"]
            cv2.line(result, p1, p2, color, 2)
            cv2.circle(result, p1, 3, color, -1)
            cv2.circle(result, p2, 3, color, -1)
        elif shape == "circle" and "radius" in f:
            cx, cy = f["centroid"]
            cv2.circle(result, (cx, cy), f["radius"], color, 2)
        else:
            cv2.rectangle(result, (x - 2, y - 2), (x + bw + 2, y + bh + 2), color, 2)

        label = f"{shape} Z{f['zone']} conf={f['confidence']:.1f} A={f['area']}"
        cv2.putText(result, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return result


def process_image(path, debug=False, out_dir=None):
    """Process a single image."""
    img = cv2.imread(path)
    if img is None:
        print(f"  ERROR: cannot read {path}")
        return

    h, w = img.shape[:2]
    base = canonical_base(path)

    if debug:
        findings, edges, contours = detect_geometry_anomalies(img, debug=True)
    else:
        findings = detect_geometry_anomalies(img, debug=False)

    print(f"\n  {os.path.basename(path)} ({w}x{h})")
    print(f"  Findings: {len(findings)}")
    for f in findings:
        extra = ""
        if "length" in f:
            extra = f" len={f['length']} ang={f.get('angle', 0)}"
        elif "radius" in f:
            extra = f" r={f['radius']}"
        elif "right_angles" in f:
            extra = f" ra={f['right_angles']} reg={f.get('regularity', 0)}"
        print(f"    {f['shape']} Z{f['zone']} area={f['area']} conf={f['confidence']} "
              f"bbox={f['bbox']}{extra}")

    # Save result
    if out_dir is None:
        out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if findings:
        result = draw_findings(img, findings)
        out_path = os.path.join(out_dir, f"{base}_geometry_anomalies.jpg")
        cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved: {out_path}")

    if debug:
        # Save edge map
        edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        blend = cv2.addWeighted(img, 0.6, edge_color, 0.4, 0)
        out_path = os.path.join(out_dir, f"{base}_edges.jpg")
        cv2.imwrite(out_path, blend, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Debug: edges -> {out_path}")

        # Save contour overlay
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        out_path = os.path.join(out_dir, f"{base}_contours.jpg")
        cv2.imwrite(out_path, contour_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Debug: contours -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Geometry anomaly detection (man-made shapes)")
    p.add_argument("output_dir",
                   help="Video output dir with report.json and scenes/ (like az-previews)")
    p.add_argument("--debug", action="store_true", help="Save debug heatmaps")
    args = p.parse_args()

    scenes_dir = os.path.join(args.output_dir, "scenes")
    anomalies_dir = os.path.join(scenes_dir, "anomalies")
    os.makedirs(anomalies_dir, exist_ok=True)

    images = collect_orig_images(scenes_dir)
    print(f"Found {len(images)} _orig.jpg images in {scenes_dir}")
    for img_path in images:
        process_image(img_path, debug=args.debug, out_dir=anomalies_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
