"""Shared utilities for SAR video analysis tools.

All derivative scripts (image.py, llm_analysis.py, previews.py,
check_all_reports.py) import from here instead of duplicating functions.

Extracted from alpine_zoom.video (source of truth).
"""
import os
import re
import json
import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────

# All enhancement variants produced by the pipeline (source of truth).
# Standalone builders previously had only 4 (missing v4); this fixes the drift.
SCENE_VARIANTS = ["_orig.jpg", "_grid_v1.jpg", "_grid_v2.jpg", "_grid_v3.jpg", "_grid_v4.jpg"]

# Variant keys in report.json (basenames) and in-memory scene dicts (full paths)
SCENE_FILE_KEYS = ["image_file", "image_file_v1", "image_file_v2", "image_file_v3", "image_file_v4"]
SCENE_PATH_KEYS = ["orig_path", "image_path_v1", "image_path_v2", "image_path_v3", "image_path_v4"]


# ── Scene filename parsing ────────────────────────────────────────────

def scene_sort_key(path):
    """Extract scene number for chronological sorting."""
    fn = os.path.basename(path)
    m = re.match(r'scene_(\d+)_f(\d+)', fn)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999999, 0)


def parse_scene_frame(path):
    """Extract scene number and frame number from filename.
    Returns (scene_num, frame_num)."""
    fn = os.path.basename(path)
    m = re.match(r'scene_(\d+)_f(\d+)', fn)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def frame_to_time(frame, fps):
    """Convert frame number to HH:MM:SS.cc string."""
    total_sec = frame / fps
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    cs = int((total_sec - int(total_sec)) * 100)
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


# ── Drawing utilities ─────────────────────────────────────────────────

def draw_text_with_bg(frame, text, x, y, font, scale, thickness, color):
    """Draw text with semi-transparent black background for readability."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


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


def draw_grid(img, zones):
    """Draw 3x3 zone grid overlay with zone labels."""
    overlay = img.copy()
    for zid, x0, y0, x1, y1 in zones:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 1)
        cv2.putText(overlay, f"Z{zid}", (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return overlay


# ── Zone findings extraction ──────────────────────────────────────────

def extract_zone_findings(scene):
    """Extract zone-to-max-confidence map from a scene's LLM findings.

    Checks both llm_fast and llm_deep results. Returns {zone_num: max_conf}
    for zones 1-9. Returns empty dict if no findings.

    This was triple-copied across alpine_zoom.video and the preview builders
    with subtle differences. Now unified.
    """
    zones_found = {}
    for llm_key in ("llm_fast", "llm_deep"):
        llm = scene.get(llm_key, {})
        if not isinstance(llm, dict) or not llm.get("objects_found"):
            continue
        scene_conf = llm.get("confidence", 0)
        for fnd in llm.get("findings", []):
            zone = fnd.get("zone", "")
            zone_num = re.search(r'(\d+)', str(zone))
            if zone_num:
                z = int(zone_num.group(1))
                if 1 <= z <= 9:
                    fnd_conf = fnd.get("confidence", scene_conf)
                    if z not in zones_found or fnd_conf > zones_found[z]:
                        zones_found[z] = fnd_conf
    return zones_found


def canonical_base(path):
    """Derive the canonical scene base (e.g. 'scene_05_f01101') from an image path.

    Strips a trailing '_orig' so anomaly output is 'scene_05_f01101_<kind>_anomalies.jpg'
    (matching az-video's naming), not '..._orig_<kind>_anomalies.jpg'.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    if base.endswith("_orig"):
        base = base[:-len("_orig")]
    return base


def collect_orig_images(scenes_dir, subfolders=("hq", "lq")):
    """Collect all _orig.jpg paths from the given subfolders of scenes_dir.

    Returns a chronologically sorted list. Used by az-colors and az-geometry
    so both scan the same scene images the same way.
    """
    images = []
    for sub in subfolders:
        d = os.path.join(scenes_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith("_orig.jpg"):
                images.append(os.path.join(d, f))
    return sorted(images, key=scene_sort_key)


def build_dynamics(findings, image_file=""):
    """Build the `dynamics` dict for a scene from its color findings.

    Single source of truth for the dynamics structure — used by both
    alpine_zoom.video (main pipeline) and alpine_zoom.colors (standalone
    az-colors --update-report) so the two never drift.
    """
    flagged_zones = list(set(f["zone"] for f in findings))
    zone_scores = {}
    for f in findings:
        z = f["zone"]
        if z not in zone_scores or f["confidence"] > zone_scores[z]:
            zone_scores[z] = f["confidence"]
    return {
        "anomaly_score": round(max((f["confidence"] for f in findings), default=0), 4),
        "zone_scores": {f"Z{k}": round(v, 4) for k, v in zone_scores.items()},
        "flagged_zones": flagged_zones,
        "color_findings": findings,
        "image_file": image_file,
    }


def build_scene_findings_map(scenes):
    """Build {scene_num: {zone_num: max_conf}} from a list of scene dicts.
    Uses scene index as the key (matching scene_id in report.json)."""
    result = {}
    for si, s in enumerate(scenes):
        zf = extract_zone_findings(s)
        if zf:
            result[si] = zf
    return result


def draw_zone_highlights(frame, zone_findings):
    """Draw red frame around zones with LLM findings + confidence score.

    Args:
        frame: OpenCV image (modified in place + returned)
        zone_findings: {zone_num: max_confidence} dict (from extract_zone_findings)
    """
    if not zone_findings:
        return frame
    h, w = frame.shape[:2]
    cell_w = w // 3
    cell_h = h // 3
    for z, conf in zone_findings.items():
        row = (z - 1) // 3
        col = (z - 1) % 3
        x0 = col * cell_w
        y0 = row * cell_h
        cv2.rectangle(frame, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (0, 0, 255), 4)
        conf_text = f"conf={conf:.1f}"
        (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = x0 + cell_w - tw - 10
        ty = y0 + th + 10
        cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, conf_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


# ── Scene image lookup ────────────────────────────────────────────────

def find_scene_image(scenes_dir, scene, variants=None):
    """Find image file for a scene, checking hq/, lq/, and scenes/ root.

    Args:
        scenes_dir: path to scenes/ directory
        scene: scene dict from report.json (uses image_file_vN keys = basenames)
               or in-memory dict (uses image_path_vN keys = full paths)
        variants: ordered list of variant suffixes to try (default: SCENE_VARIANTS)

    Returns: path to first existing image, or None.
    """
    if variants is None:
        variants = SCENE_VARIANTS

    # Try full-path keys first (in-memory scene dict from pipeline)
    for path_key in SCENE_PATH_KEYS:
        path = scene.get(path_key, "")
        if path and os.path.exists(path):
            return path

    # Fall back to basename keys (report.json) — search in hq/, lq/, scenes/
    # Map variant suffixes to report.json keys
    suffix_to_key = {
        "_orig.jpg": "image_file",  # orig is stored as image_file in some reports
        "_grid_v1.jpg": "image_file_v1",
        "_grid_v2.jpg": "image_file_v2",
        "_grid_v3.jpg": "image_file_v3",
        "_grid_v4.jpg": "image_file_v4",
    }
    # Also try image_file as v3 alias (common in report.json)
    for suffix in variants:
        key = suffix_to_key.get(suffix, "")
        if not key:
            continue
        fname = scene.get(key, "")
        if not fname:
            # Try constructing from scene_id + best_frame
            si = scene.get("scene_id", scene.get("scene_num", None))
            fi = scene.get("best_frame", None)
            if si is not None and fi is not None:
                base = f"scene_{si:02d}_f{fi:05d}"
                fname = base + suffix
        if not fname:
            continue
        for sub in ("hq", "lq", ""):
            path = os.path.join(scenes_dir, sub, fname) if sub else os.path.join(scenes_dir, fname)
            if os.path.exists(path):
                return path
    return None


def find_all_variants(scenes_dir, scene):
    """Find all variant images for a scene.

    Returns list of (label, path) tuples for existing variants.
    """
    variants = []
    label_map = {
        "image_file": ("orig", "_orig.jpg"),
        "image_file_v1": ("v1_high_contrast", "_grid_v1.jpg"),
        "image_file_v2": ("v2_gentle_clahe", "_grid_v2.jpg"),
        "image_file_v3": ("v3_aggressive_shadow", "_grid_v3.jpg"),
    }
    # Also check path keys (in-memory)
    path_label_map = {
        "orig_path": "orig",
        "image_path_v1": "v1_high_contrast",
        "image_path_v2": "v2_gentle_clahe",
        "image_path_v3": "v3_aggressive_shadow",
    }

    for key, (label, suffix) in label_map.items():
        fname = scene.get(key, "")
        if not fname:
            # Try constructing from scene_id + best_frame
            si = scene.get("scene_id", scene.get("scene_num", None))
            fi = scene.get("best_frame", None)
            if si is not None and fi is not None:
                base = f"scene_{si:02d}_f{fi:05d}"
                fname = base + suffix
        if not fname:
            continue
        for sub in ("hq", "lq", ""):
            path = os.path.join(scenes_dir, sub, fname) if sub else os.path.join(scenes_dir, fname)
            if os.path.exists(path):
                variants.append((label, path))
                break

    return variants


# ── Report loading ────────────────────────────────────────────────────

def load_report(output_dir):
    """Load report.json from an output directory.

    Returns (report_dict, scenes_dir) or raises FileNotFoundError.
    """
    report_path = os.path.join(output_dir, "report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"No report.json in {output_dir}")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    scenes_dir = os.path.join(output_dir, "scenes")
    return report, scenes_dir


def find_video_dirs(root):
    """Find all dirs containing report.json under root."""
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "report.json" in filenames:
            result.append(dirpath)
    result.sort()
    return result
