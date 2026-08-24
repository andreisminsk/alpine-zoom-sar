"""
SAR Scene Analyzer — LLM-first approach.

Strategy:
  1. Scan all frames, score quality (sharpness, motion blur, exposure, zoom)
  2. Group high-quality frames into distinct scenes (by visual similarity)
  3. Pick best frame per scene
  4. Send to vision LLM with SAR guidance + known jacket colors:
     - dark (black/dark gray)
     - blue
     - orange
  5. LLM analyzes full frame, reports findings
  6. Human review queue with annotated results

No CV detectors — LLM does all visual analysis on pre-filtered quality frames.
"""
import sys
import os
import json
import time
import re
import argparse
import base64
import urllib.request
import urllib.error
import numpy as np
import cv2
from PIL import Image
import piexif

sys.stdout.reconfigure(encoding="utf-8")

# ── Shared modules (extracted from this file) ─────────────────────────
# These were originally defined inline; now imported from shared modules
# to keep all derivative scripts in sync.
from alpine_zoom.common import (
    scene_sort_key,
    draw_text_with_bg,
    zone_grid,
    draw_grid,
    extract_zone_findings,
    build_scene_findings_map,
    build_dynamics,
    draw_zone_highlights,
    find_scene_image,
    find_all_variants,
    SCENE_VARIANTS,
)
from alpine_zoom.llm import (
    FOUNDATIONAL_PROMPT,
    SAR_PROMPT,
    MissionContext,
    build_prompt,
    encode_image,
    llm_analyze,
    llm_text_analyze,
    two_stage_analyze,
    parallel_llm_batch,
    get_llm_variants,
    get_context,
    default_sar_context,
    default_sar_heli_context,
    VISION_PROMPT,
    build_reasoning_prompt,
)
from alpine_zoom.previews import (
    build_llm_findings_preview,
    build_hq_lq_previews,
    build_color_anomalies_preview,
    build_contact_sheet,
)
from alpine_zoom.color import (
    detect_color_anomalies,
    draw_color_findings,
    estimate_color as _estimate_color,
)
def _gps_to_exif_rational(value):
    """Convert decimal degrees to EXIF rational (degrees, minutes, seconds).
    Returns tuple of 3 (numerator, denominator) pairs as piexif expects."""
    deg = int(abs(value))
    min_rem = (abs(value) - deg) * 60
    minutes = int(min_rem)
    sec = (min_rem - minutes) * 60
    return ((deg, 1), (minutes, 1), (int(sec * 10000), 10000))


def save_with_exif(cv_img, path, gps=None, scene_num=0, frame_num=0, video_fn=""):
    """Save OpenCV image as JPEG with EXIF GPS + metadata embedded via piexif."""
    if gps is None:
        cv2.imwrite(path, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    lat = gps.get("lat", 0)
    lon = gps.get("lon", 0)
    abs_alt = gps.get("abs_alt", 0)
    rel_alt = gps.get("rel_alt", 0)

    # Build ImageDescription with all metadata
    desc_parts = [f"scene:{scene_num}", f"frame:{frame_num}", f"video:{video_fn}"]
    if "yaw" in gps:
        desc_parts.append(f"yaw:{gps['yaw']:.1f}")
    if "pitch" in gps:
        desc_parts.append(f"pitch:{gps['pitch']:.1f}")
    if "focal_len" in gps:
        desc_parts.append(f"focal_len:{gps['focal_len']:.1f}")
    if "dzoom_ratio" in gps:
        desc_parts.append(f"zoom:{gps['dzoom_ratio']:.1f}")
    if "rel_alt" in gps:
        desc_parts.append(f"rel_alt:{rel_alt:.1f}")
    if "abs_alt" in gps:
        desc_parts.append(f"abs_alt:{abs_alt:.1f}")
    description = " | ".join(desc_parts)

    # Build DateTime from SRT
    dt_str = ""
    if "datetime" in gps:
        parts = gps["datetime"].split(" ")
        dt_str = parts[0].replace("-", ":") + " " + parts[1]

    # Build EXIF using piexif
    zeroth_ifd = {
        piexif.ImageIFD.ImageDescription: description,
    }
    if dt_str:
        zeroth_ifd[piexif.ImageIFD.DateTime] = dt_str

    exif_ifd = {}
    if dt_str:
        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = dt_str
        exif_ifd[piexif.ExifIFD.DateTimeDigitized] = dt_str
    if "focal_len" in gps:
        exif_ifd[piexif.ExifIFD.FocalLength] = (int(gps["focal_len"] * 100), 100)
    if "dzoom_ratio" in gps:
        exif_ifd[piexif.ExifIFD.DigitalZoomRatio] = (int(gps["dzoom_ratio"] * 100), 100)

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: _gps_to_exif_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: _gps_to_exif_rational(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSAltitude: (int(abs(abs_alt * 100)), 100),
    }

    exif_dict = {
        "0th": zeroth_ifd,
        "Exif": exif_ifd,
        "GPS": gps_ifd,
    }
    exif_bytes = piexif.dump(exif_dict)

    # Save with PIL + piexif bytes
    pil_img.save(path, "JPEG", quality=95, exif=exif_bytes)


def parse_srt_gps(srt_path):
    """Parse DJI SRT subtitle file for per-frame GPS coordinates.

    Returns dict: {frame_num: {"lat": float, "lon": float, "rel_alt": float, "abs_alt": float}}
    """
    if not srt_path or not os.path.exists(srt_path):
        return {}

    gps_data = {}
    try:
        with open(srt_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split into subtitle blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            # Extract FrameCnt
            fc_match = re.search(r'FrameCnt:\s*(\d+)', block)
            if not fc_match:
                continue
            frame_num = int(fc_match.group(1))

            # Extract all available telemetry fields
            entry = {}

            # Timestamp from SRT (e.g. "2026-08-15 19:49:39.873")
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', block)
            if ts_match:
                entry["date"] = ts_match.group(1)
                entry["time"] = ts_match.group(2)
                entry["datetime"] = f"{ts_match.group(1)} {ts_match.group(2)}"

            # GPS + altitude
            for field, key in [("latitude", "lat"), ("longitude", "lon"),
                               ("rel_alt", "rel_alt"), ("abs_alt", "abs_alt")]:
                m = re.search(rf'{field}:\s*(-?[\d.]+)', block)
                if m:
                    entry[key] = float(m.group(1))

            # Gimbal orientation
            for field, key in [("gb_yaw", "yaw"), ("gb_pitch", "pitch"),
                               ("gb_roll", "roll")]:
                m = re.search(rf'{field}:\s*(-?[\d.]+)', block)
                if m:
                    entry[key] = float(m.group(1))

            # Camera settings
            for field, key in [("focal_len", "focal_len"), ("dzoom_ratio", "dzoom_ratio")]:
                m = re.search(rf'{field}:\s*(-?[\d.]+)', block)
                if m:
                    entry[key] = float(m.group(1))

            # Distance to ground point (some firmware versions)
            m = re.search(r'distance:\s*(-?[\d.]+)', block)
            if m:
                entry["distance"] = float(m.group(1))

            # Drone speed (some firmware versions)
            m = re.search(r'speed:\s*(-?[\d.]+)', block)
            if m:
                entry["speed"] = float(m.group(1))

            if entry:
                gps_data[frame_num] = entry
    except Exception as e:
        print(f"  WARNING: failed to parse SRT: {e}")

    return gps_data


def find_srt_for_video(video_path):
    """Find companion .SRT file for a video."""
    base, _ = os.path.splitext(video_path)
    srt_path = base + ".SRT"
    if os.path.exists(srt_path):
        return srt_path
    # Try lowercase
    srt_path = base + ".srt"
    if os.path.exists(srt_path):
        return srt_path
    return None


def get_gps_for_frame(gps_data, frame_num):
    """Get GPS for a frame, falling back to nearest available frame."""
    if not gps_data:
        return None
    if frame_num in gps_data:
        return gps_data[frame_num]
    # Find nearest frame with GPS data
    nearest = min(gps_data.keys(), key=lambda k: abs(k - frame_num))
    if abs(nearest - frame_num) <= 30:  # within 1 second
        return gps_data[nearest]
    return None


# ── Frame quality ──────────────────────────────────────────────────────

def frame_quality(gray, prev_gray=None, helicopter=False):
    """Score frame quality 0-1. Returns (score, reasons).
    
    helicopter=True relaxes thresholds — helicopter footage has more 
    vibration, motion blur, and frame-to-frame shift, but still contains
    valuable exterior terrain data.
    """
    reasons = []
    score = 1.0

    if helicopter:
        # Relaxed thresholds for helicopter footage (but not too relaxed)
        blur_severe = 30    # was 20, was 50 for drone
        blur_soft = 60      # was 40, was 100 for drone
        blur_ok = 120        # was 80, was 200 for drone
        shift_fast = 60     # was 30
        shift_mod = 35       # was 15
        dark_thresh = 10    # was 20
        bright_thresh = 245  # was 240
        contrast_thresh = 12 # was 20
    else:
        blur_severe = 50
        blur_soft = 100
        blur_ok = 200
        shift_fast = 30
        shift_mod = 15
        dark_thresh = 20
        bright_thresh = 240
        contrast_thresh = 20

    # Sharpness
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap < blur_severe:
        score *= 0.3
        reasons.append(f"blurry({lap:.0f})")
    elif lap < blur_soft:
        score *= 0.6
        reasons.append(f"soft({lap:.0f})")
    elif lap < blur_ok:
        score *= 0.85
        reasons.append(f"ok({lap:.0f})")

    # Motion / shift
    if prev_gray is not None:
        try:
            shift, resp = cv2.phaseCorrelate(
                prev_gray.astype(np.float64), gray.astype(np.float64))
            mag = np.hypot(shift[0], shift[1])
            if mag > shift_fast:
                score *= 0.5
                reasons.append(f"fast_move({mag:.0f}px)")
            elif mag > shift_mod:
                score *= 0.75
                reasons.append(f"move({mag:.0f}px)")
        except Exception:
            pass

    # Exposure
    mean_b = np.mean(gray)
    if mean_b < dark_thresh:
        score *= 0.2
        reasons.append("dark")
    elif mean_b > bright_thresh:
        score *= 0.4
        reasons.append("overexposed")
    elif mean_b > 220:
        score *= 0.8
        reasons.append("bright")

    # Contrast — low contrast = haze/flat
    std = np.std(gray)
    if std < contrast_thresh:
        score *= 0.5
        reasons.append(f"low_contrast({std:.0f})")

    return score, reasons


# ── Scene grouping ─────────────────────────────────────────────────────

def frame_signature(gray, size=16):
    """Downsample to small grayscale image for scene comparison.
    16x16 is coarse enough to tolerate drone movement."""
    small = cv2.resize(gray, (size, size))
    return small.astype(np.float32).flatten()


def scene_similarity(sig_a, sig_b):
    """Normalized cross-correlation of signatures. 0=different, 1=identical."""
    a = sig_a - np.mean(sig_a)
    b = sig_b - np.mean(sig_b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def group_into_scenes(quality_frames, similarity_thresh=0.85, max_frame_gap=300):
    """Group frames into scenes by visual similarity.
    Only matches against scenes within max_frame_gap frames — prevents
    similar terrain at different times from being grouped together.
    quality_frames: list of (frame_idx, gray, score, reasons, signature)
    Returns list of scenes: [{frame_indices, best_frame, best_score, ...}]
    """
    scenes = []
    for fi, gray, score, reasons, sig in quality_frames:
        # Find matching scene (only recent ones — timeline-local grouping)
        matched = False
        for scene in scenes:
            # Skip scenes that are too far back in timeline
            last_frame = scene["frames"][-1][0]
            if fi - last_frame > max_frame_gap:
                continue
            # Compare to scene's representative signature
            rep_sig = scene["rep_signature"]
            sim = scene_similarity(sig, rep_sig)
            if sim >= similarity_thresh:
                scene["frames"].append((fi, score, reasons))
                scene["signatures"].append(sig)
                if score > scene["best_score"]:
                    scene["best_score"] = score
                    scene["best_frame"] = fi
                    scene["best_reasons"] = reasons
                    scene["rep_signature"] = sig
                matched = True
                break
        if not matched:
            scenes.append({
                "frames": [(fi, score, reasons)],
                "signatures": [sig],
                "best_frame": fi,
                "best_score": score,
                "best_reasons": reasons,
                "rep_signature": sig,
            })
    return scenes


# ── Scene deduplication (PASS 2.5) ─────────────────────────────────────

DEDUP_COLOR_THRESH = 0.85    # Color histogram intersection (L1-normalized)
DEDUP_COMBINED_THRESH = 0.85 # Same as color — histogram-only dedup


def scene_signature_rich(gray, color_img=None):
    """Compute color histogram signature for dedup.
    Uses 8×8×8 HSV histogram (L1-normalized) — shift-invariant, works on
    high-vibration helicopter footage where spatial signatures fail.
    Returns (None, color_hist) or (None, None) if no color.
    """
    if color_img is not None:
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        hist = hist / max(hist.sum(), 1)  # L1 normalize (sum=1)
        return None, hist.flatten()
    return None, None


def dedup_similarity(sig_a, sig_b, hist_a, hist_b):
    """Compute similarity between two scenes using color histogram intersection.
    Returns (combined, struct_sim, color_sim).
    """
    struct_sim = 0.0  # Not used — spatial signatures fail on vibrating footage
    color_sim = 0.0
    if hist_a is not None and hist_b is not None:
        color_sim = float(np.minimum(hist_a, hist_b).sum())
    combined = color_sim  # Histogram-only
    return combined, struct_sim, color_sim


def dedup_scenes(scenes, video_path, combined_thresh=DEDUP_COMBINED_THRESH):
    """Merge near-duplicate scenes (PASS 2.5).

    Compares all scenes pairwise using 32×32 grayscale + color histograms.
    Merges scenes with combined similarity above threshold, regardless of
    temporal distance (catches camera returning to same viewpoint).

    Returns (deduplicated_scenes, n_merged) where scenes are renumbered
    sequentially and each has a 'merged_from' field listing original IDs.
    """
    if len(scenes) <= 1:
        for i, s in enumerate(scenes):
            s["scene_id"] = i
            s["merged_from"] = [i]
        return scenes, 0

    # Compute rich signatures for each scene's best frame
    cap = cv2.VideoCapture(video_path)
    rich_sigs = []
    for scene in scenes:
        fi = scene["best_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            rich_sigs.append((None, None))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        struct, color = scene_signature_rich(gray, frame)
        rich_sigs.append((struct, color))
    cap.release()

    # Greedy clustering: assign each scene to an existing cluster or start new one
    clusters = []  # list of list of scene indices
    for i in range(len(scenes)):
        if rich_sigs[i][1] is None:
            clusters.append([i])
            continue
        best_cluster = -1
        best_sim = combined_thresh
        for ci, cluster in enumerate(clusters):
            # Compare to the first scene in the cluster (representative)
            j = cluster[0]
            if rich_sigs[j][1] is None:
                continue
            combined, _, color_sim = dedup_similarity(
                rich_sigs[i][0], rich_sigs[j][0],
                rich_sigs[i][1], rich_sigs[j][1])
            if color_sim >= DEDUP_COLOR_THRESH and combined > best_sim:
                best_sim = combined
                best_cluster = ci
        if best_cluster >= 0:
            clusters[best_cluster].append(i)
        else:
            clusters.append([i])

    # Build merged scenes
    merged_scenes = []
    n_merged = 0
    for cluster in clusters:
        # Pick representative: highest best_score, tiebreak most frames
        best_idx = max(cluster, key=lambda i: (scenes[i]["best_score"],
                                                len(scenes[i]["frames"])))
        rep = scenes[best_idx].copy()
        # Merge frames from all cluster members
        all_frames = []
        all_sigs = []
        for i in cluster:
            all_frames.extend(scenes[i]["frames"])
            all_sigs.extend(scenes[i]["signatures"])
        rep["frames"] = all_frames
        rep["signatures"] = all_sigs
        rep["merged_from"] = sorted(cluster)
        if len(cluster) > 1:
            n_merged += len(cluster) - 1
        merged_scenes.append(rep)

    # Sort chronologically and renumber
    merged_scenes.sort(key=lambda s: s["best_frame"])
    for i, s in enumerate(merged_scenes):
        s["scene_id"] = i

    return merged_scenes, n_merged


# ── Zone grid ──────────────────────────────────────────────────────────

# ── Enhancement ────────────────────────────────────────────────────────

def enhance_frame_v1(img):
    """Original high-contrast enhancement: saturation boost + histogram equalization.
    Best for: bright areas, color spotting (orange/blue jackets on snow).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.8, 0, 255).astype(np.uint8)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def enhance_frame_v2(img):
    """Shadow-recovery enhancement: gentle gamma lift + CLAHE + saturation boost.
    Best for: dark areas, shadows, caves, dark rock — reveals hidden details
    without amplifying noise or washing out midtones.
    """
    # Stage 1: Gentle shadow recovery via gamma on L channel of LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)
    l_norm = l_chan / 255.0
    l_gamma = np.power(l_norm, 0.7) * 255.0  # gamma=0.7, gentle lift
    lab[:, :, 0] = np.clip(l_gamma, 0, 255).astype(np.uint8)
    shadow_lifted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Stage 2: CLAHE on L channel — gentle local contrast
    lab2 = cv2.cvtColor(shadow_lifted, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab2[:, :, 0] = clahe.apply(lab2[:, :, 0])
    contrast_enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Stage 3: Saturation boost only (no histogram equalization — avoids noise)
    hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.6, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def enhance_frame_v3(img):
    """Aggressive shadow recovery: shadow mask lift 130 + CLAHE + saturation.
    Best for: maximum shadow detail — rescue priority, see if someone is hidden.
    No denoise — preserves every detail.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    shadow_mask = np.clip((120.0 - l) / 120.0, 0, 1)
    shadow_mask_smooth = cv2.GaussianBlur(shadow_mask, (31, 31), 0)
    l_lifted = l + shadow_mask_smooth * 130.0
    lab[:, :, 0] = np.clip(l_lifted, 0, 255).astype(np.uint8)
    shadow_lifted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    lab2 = cv2.cvtColor(shadow_lifted, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab2[:, :, 0] = clahe.apply(lab2[:, :, 0])
    contrast = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.6, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def enhance_frame_v4(img):
    """Highlight recovery: detect blown highlights, pull down, apply CLAHE.
    Best for: overexposed snow, blown-out highlights — recovers snow texture.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)

    # Highlight mask: pixels where L > 220 (near-white)
    highlight_mask = np.clip((l - 220) / (255 - 220), 0, 1)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (51, 51), 0)

    # Pull down highlights: reduce L in bright areas
    l_reduced = l - highlight_mask * 40
    l_reduced = np.clip(l_reduced, 0, 255)

    lab_mod = lab.copy()
    lab_mod[:, :, 0] = l_reduced.astype(np.uint8)
    mod_bgr = cv2.cvtColor(lab_mod, cv2.COLOR_LAB2BGR)

    # CLAHE on the modified image
    lab2 = cv2.cvtColor(mod_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab2[:, :, 0] = clahe.apply(lab2[:, :, 0])
    result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Blend: use CLAHE result only in highlight areas
    mask_3ch = cv2.merge([highlight_mask, highlight_mask, highlight_mask])
    result = np.where(mask_3ch > 0.3, result, img)

    return result


def enhance_frame(img):
    """Default enhancement — uses v3 (aggressive shadow recovery)."""
    return enhance_frame_v3(img)


# ── LLM verification ───────────────────────────────────────────────────
# LLM functions (build_prompt, encode_image, llm_analyze, llm_text_analyze,
# two_stage_analyze, get_llm_variants) are imported from alpine_zoom.llm above.
# Prompt system (FOUNDATIONAL_PROMPT, VISION_PROMPT, build_reasoning_prompt,
# MissionContext, presets) is imported from alpine_zoom.llm / alpine_zoom.context above.


def extract_recording_time(video_path, gps_data=None):
    """Extract recording time from SRT GPS data, companion XML, filename, or video EXIF.
    Returns ISO datetime string or None."""
    # 1. SRT GPS data (most reliable — per-frame telemetry)
    if gps_data:
        for fi in sorted(gps_data.keys())[:5]:
            dt = gps_data[fi].get("datetime")
            if dt:
                return dt
    # 2. Companion XML metadata file (DJI/Sony professional cameras)
    base, _ = os.path.splitext(video_path)
    for ext in ("M01.XML", "M01.xml", "XML", "xml"):
        xml_path = base + ext
        if os.path.exists(xml_path):
            try:
                with open(xml_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                m = re.search(r'CreationDate\s+value="([^"]+)"', content)
                if m:
                    return m.group(1)
            except Exception:
                pass
    # 3. Filename pattern: DJI_YYYYMMDDHHMMSS_NNNN_Z.MP4 or similar
    fn = os.path.basename(video_path)
    m = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', fn)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}"
    # 4. Video file EXIF/metadata via OpenCV (limited support)
    try:
        cap = cv2.VideoCapture(video_path)
        cap.release()
    except Exception:
        pass
    return None


# ── Color anomaly detection (scene-relative) ──────────────────────────

# ── Color anomaly detection ───────────────────────────────────────────
# detect_color_anomalies, draw_color_findings, estimate_color are imported
# from alpine_zoom.color above.


# ── Main pipeline ──────────────────────────────────────────────────────


# ── Main pipeline ──────────────────────────────────────────────────────

def analyze_video(video_path, out_dir, quality_thresh=0.5,
                  scene_sim_thresh=0.65, stride=10,
                  fast_model="gemma4:31b-cloud",
                  deep_model="qwen3.5:397b-cloud",
                  deep_top_n=50, llm_scenes_cap=50, helicopter=False,
                  llm_pipeline="fast", stride_mode="dynamic",
                  from_sec=None, to_sec=None, recording_time_override=None,
                  color_anomalies=False, llm_run=False, build_preview=False,
                  dedup_thresh=DEDUP_COMBINED_THRESH, mission_context=None,
                  llm_two_stage=False, llm_reasoning_model="glm-5.1:cloud",
                  llm_parallel=0):
    analysis_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    scenes_dir = os.path.join(out_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames, {total/fps:.0f}s")
    print(f"Stride: every {stride} frames")
    print(f"Quality threshold: {quality_thresh}")
    print(f"Scene similarity threshold: {scene_sim_thresh}")

    zones = zone_grid(h, w)

    # Parse companion SRT file for GPS data
    srt_path = find_srt_for_video(video_path)
    gps_data = {}
    if srt_path:
        gps_data = parse_srt_gps(srt_path)
        print(f"GPS data: {len(gps_data)} frames from SRT: {os.path.basename(srt_path)}")
    else:
        print("GPS data: no SRT file found")

    # Extract recording time for absolute UTC timestamps
    if recording_time_override:
        recording_time = recording_time_override
        print(f"Recording time: {recording_time} (manual override)")
    else:
        recording_time = extract_recording_time(video_path, gps_data)
        if recording_time:
            print(f"Recording time: {recording_time}")
        else:
            print("Recording time: not available")


    # Build prompt (helicopter mode adds interior-ignore instructions)
    # Derive pipeline mode flags from --llm-pipeline
    llm_pipeline_full = (llm_pipeline == "max")
    llm_pipeline_chancepeek = (llm_pipeline == "chancepeek")

    sar_prompt = build_prompt(helicopter=helicopter, mission_context=mission_context)
    if mission_context:
        ctx_name = mission_context.name or 'custom'
        if args.context_file:
            print(f"Mission context: {ctx_name} (from --context-file)")
        elif args.context_preset:
            print(f"Mission context: {ctx_name} (from --context-preset)")
        elif args.helicopter:
            print(f"Mission context: {ctx_name} (auto from --helicopter)")
        else:
            print(f"Mission context: {ctx_name} (default)")
    else:
        print("Mission context: none (foundational prompt only)")
    if helicopter:
        print("HELICOPTER MODE: LLM will ignore interior passengers/elements")
        # Lower quality threshold — helicopter footage is lower quality but still valuable
        if quality_thresh > 0.4:
            print(f"  Lowering quality threshold: {quality_thresh} → 0.4")
            quality_thresh = 0.4
        # More aggressive scene grouping — helicopter moves fast, many "distinct" scenes
        if scene_sim_thresh > 0.55:
            print(f"  Lowering scene similarity: {scene_sim_thresh} → 0.55 (fewer, bigger scenes)")
            scene_sim_thresh = 0.55
        # Analyze more scenes — can't afford to miss findings
        if llm_scenes_cap != "all" and isinstance(llm_scenes_cap, int) and llm_scenes_cap < 100:
            print(f"  Increasing LLM scenes cap: {llm_scenes_cap} → 100")
            llm_scenes_cap = 100

    # ── Compute frame range from --from/--to ──────────────────────────
    from_frame = int(from_sec * fps) if from_sec is not None else 0
    to_frame = int(to_sec * fps) if to_sec is not None else total
    if from_sec is not None or to_sec is not None:
        print(f"Processing range: {from_frame}–{to_frame} frames "
              f"({from_sec or 0:.1f}s – {to_sec or total/fps:.1f}s)")

    # ── PASS 1: Scan quality ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASS 1: Scanning frame quality (stride: {stride_mode})")
    print(f"{'='*60}")

    # Determine stride mode
    is_dynamic = isinstance(stride_mode, str) and stride_mode.lower() == "dynamic"
    fixed_stride = 1 if is_dynamic else int(stride_mode)

    # Dynamic stride parameters
    DYN_MIN_STRIDE = 1       # Densest sampling — every frame (fast motion/zoom)
    DYN_MAX_STRIDE = 20      # Sparsest sampling (slow pan, static)
    DYN_FAST_THRESH = 20     # px shift → fast motion (lower = catch zoom earlier)
    DYN_MODERATE_THRESH = 8  # px shift → moderate motion
    DYN_ZOOM_SPIKE = 60      # px shift → zoom event, force immediate sample
    dyn_current_stride = 10  # Start at default
    dyn_frames_since_last = 0  # Frames since last sampled frame
    dyn_force_sample = False   # Force sampling on next frame

    quality_frames = []
    gray_prev = None
    frame_idx = 0
    t0 = time.time()

    # Seek to start frame if needed
    if from_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)
        frame_idx = from_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= to_frame:
            break

        # In dynamic mode: read every frame, compute motion, decide whether to sample
        # In fixed mode: sample every Nth frame
        should_sample = False
        if is_dynamic:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray_prev is not None:
                try:
                    shift, resp = cv2.phaseCorrelate(
                        gray_prev.astype(np.float64), gray.astype(np.float64))
                    mag = np.hypot(shift[0], shift[1])
                except Exception:
                    mag = 0
                # Adapt stride based on motion
                if mag > DYN_ZOOM_SPIKE:
                    # Zoom spike: force immediate sample + densest stride
                    dyn_current_stride = DYN_MIN_STRIDE
                    dyn_force_sample = True
                elif mag > DYN_FAST_THRESH:
                    dyn_current_stride = DYN_MIN_STRIDE
                elif mag > DYN_MODERATE_THRESH:
                    dyn_current_stride = max(DYN_MIN_STRIDE, min(DYN_MAX_STRIDE, dyn_current_stride - 1))
                else:
                    dyn_current_stride = min(DYN_MAX_STRIDE, dyn_current_stride + 1)
            else:
                mag = 0
            dyn_frames_since_last += 1
            should_sample = dyn_frames_since_last >= dyn_current_stride or dyn_force_sample
            if should_sample:
                dyn_force_sample = False
        else:
            should_sample = frame_idx % fixed_stride == 0
            if should_sample:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if should_sample:
            if is_dynamic:
                dyn_frames_since_last = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if not is_dynamic else gray
            score, reasons = frame_quality(gray, gray_prev, helicopter=helicopter)
            timestamp = frame_idx / fps

            if score >= quality_thresh:
                sig = frame_signature(gray)
                quality_frames.append((frame_idx, gray, score, reasons, sig))
                tag = "OK"
            else:
                tag = "SKIP"

            elapsed = time.time() - t0
            motion_info = f" move={mag:.0f}px s={dyn_current_stride}" if is_dynamic else ""
            print(f"  f{frame_idx:5d} t={timestamp:6.1f}s q={score:.2f} {tag} "
                  f"{' '.join(reasons)}{motion_info} [{elapsed:.0f}s]")

            gray_prev = gray

        frame_idx += 1

    cap.release()
    scanned = frame_idx // (fixed_stride if not is_dynamic else 1)
    print(f"\nQuality scan: {len(quality_frames)} frames passed "
          f"(threshold={quality_thresh}, mode={stride_mode})")

    # ── PASS 2: Group into scenes ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("PASS 2: Grouping into distinct scenes")
    print(f"{'='*60}")

    scenes = group_into_scenes(quality_frames, scene_sim_thresh)
    print(f"Found {len(scenes)} distinct scenes from {len(quality_frames)} quality frames")

    # Sort scenes chronologically (by frame number)
    scenes.sort(key=lambda s: s["best_frame"])

    # ── PASS 2.5: Deduplicate near-duplicate scenes ──────────────────
    if dedup_thresh > 0:
        print(f"\n{'='*60}")
        print("PASS 2.5: Deduplicating near-duplicate scenes")
        print(f"{'='*60}")
        n_before = len(scenes)
        scenes, n_merged = dedup_scenes(scenes, video_path, combined_thresh=dedup_thresh)
        print(f"  Merged {n_merged} duplicate scenes ({n_before} → {len(scenes)})")
        for s in scenes:
            if len(s["merged_from"]) > 1:
                print(f"  Scene {s['scene_id']:02d} f{s['best_frame']:05d}: "
                      f"merged from {s['merged_from']}")
    else:
        for i, s in enumerate(scenes):
            s["scene_id"] = i
            s["merged_from"] = [i]

    # ── Determine which scenes to analyze (before saving images) ──────
    # Minimum quality for LLM eligibility — low-quality frames waste LLM budget
    LLM_MIN_QUALITY = 0.4

    cap = len(scenes) if llm_scenes_cap == "all" else int(llm_scenes_cap)
    if len(scenes) > cap:
        print(f"\nCapping {len(scenes)} scenes to top {cap} by quality + temporal spread")
        sorted_by_q = sorted(range(len(scenes)), key=lambda i: -scenes[i]["best_score"])
        top_candidates = sorted_by_q[:cap * 2]
        top_candidates.sort(key=lambda i: scenes[i]["best_frame"])
        # Evenly spread across the timeline — use linspace to avoid
        # step=1 truncating to first N scenes
        n = len(top_candidates)
        if n > cap:
            indices = [int(i * (n - 1) / (cap - 1)) for i in range(cap)]
            selected = [top_candidates[i] for i in indices]
        else:
            selected = top_candidates
        analyze_set = set(selected)
        print(f"  Selected {len(analyze_set)} scenes for LLM (hq), rest saved as lq")

        # ── Budget warnings ──────────────────────────────────────────
        video_dur = total / fps
        if cap < 20:
            pct = round(cap / len(scenes) * 100)
            print(f"  ⚠ Low scene cap ({cap}) for {len(scenes)} scenes — only {pct}% of video will be analyzed.")
            print(f"    Consider --llm-scenes-cap all or a higher value.")
        if video_dur > 600 and cap < 30:
            interval = round(video_dur / cap)
            print(f"  ⚠ Long video ({video_dur/60:.0f}min) with low scene cap ({cap}) — ~1 scene per {interval}s.")
    else:
        analyze_set = set(range(len(scenes)))

    # ── Create hq/lq subfolders ───────────────────────────────────────
    hq_dir = os.path.join(scenes_dir, "hq")
    lq_dir = os.path.join(scenes_dir, "lq")
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    # ── PASS 3: Extract best frame per scene + enhance ────────────────
    print(f"\n{'='*60}")
    print("PASS 3: Extracting best frame per scene")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    for si, scene in enumerate(scenes):
        fi = scene["best_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = fi / fps
        scene["timestamp"] = round(timestamp, 2)
        scene["time_str"] = f"{int(timestamp//60):02d}:{timestamp%60:05.2f}"
        scene["n_frames_in_scene"] = len(scene["frames"])
        scene["frame_range"] = [scene["frames"][0][0], scene["frames"][-1][0]]

        # Save 4 files: original, v1 high-contrast+grid, v2 gentle-clahe+grid, v3 aggressive-shadow+grid
        # HQ scenes (sent to LLM) go to hq/, the rest to lq/
        base = f"scene_{si:02d}_f{fi:05d}"
        scene_dir = hq_dir if si in analyze_set else lq_dir
        orig_path = os.path.join(scene_dir, f"{base}_orig.jpg")
        grid_v1_path = os.path.join(scene_dir, f"{base}_grid_v1.jpg")
        grid_v2_path = os.path.join(scene_dir, f"{base}_grid_v2.jpg")
        grid_v3_path = os.path.join(scene_dir, f"{base}_grid_v3.jpg")
        grid_v4_path = os.path.join(scene_dir, f"{base}_grid_v4.jpg")

        # Build text markers (video filename + GPS left, scene/frame/time right)
        video_fn = os.path.basename(video_path)
        h_val = int(timestamp // 3600)
        m_val = int((timestamp % 3600) // 60)
        s_val = int(timestamp % 60)
        cs_val = int((timestamp - int(timestamp)) * 100)
        time_str_full = f"{h_val:02d}:{m_val:02d}:{s_val:02d}.{cs_val:02d}"
        right_text_1 = f"scene: {si:04d}, frame: {fi:05d}"
        # Compute absolute UTC time if recording time available
        abs_time_str = "n/a"
        if recording_time:
            try:
                from datetime import datetime, timedelta, timezone
                # Parse recording time (handle ISO with Z or space separator)
                rt = recording_time.replace("T", " ").replace("Z", "")
                rt_dt = datetime.strptime(rt[:19], "%Y-%m-%d %H:%M:%S")
                if recording_time.endswith("Z"):
                    rt_dt = rt_dt.replace(tzinfo=timezone.utc)
                abs_dt = rt_dt + timedelta(seconds=timestamp)
                abs_time_str = abs_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                abs_time_str = "n/a"
        right_text_2 = f"rel_time: {time_str_full}, abs_time: {abs_time_str}"
        marker_font = cv2.FONT_HERSHEY_SIMPLEX

        # Get GPS for this frame
        gps = get_gps_for_frame(gps_data, fi)
        gps_text = ""
        gps_text2 = ""
        gps_text3 = ""
        if gps:
            lat = gps.get("lat", 0)
            lon = gps.get("lon", 0)
            rel_alt = gps.get("rel_alt", 0)
            abs_alt = gps.get("abs_alt", 0)
            gps_text = f"GPS: {lat:.6f}, {lon:.6f}"
            gps_text3 = f"abs_alt:{abs_alt:.0f}m, rel_alt:{rel_alt:.0f}m"

            # Ground distance removed — not meaningful enough without full context

            # Second line: yaw, pitch, focal, zoom, ground_dist, distance, speed
            parts = []
            if "yaw" in gps:
                parts.append(f"yaw:{gps['yaw']:.0f}")
            if "pitch" in gps:
                parts.append(f"pitch:{gps['pitch']:.0f}")
            if "focal_len" in gps:
                parts.append(f"fl:{gps['focal_len']:.0f}mm")
            if "dzoom_ratio" in gps:
                parts.append(f"zoom:{gps['dzoom_ratio']:.1f}x")
            if "ground_dist" in gps:
                parts.append(f"gnd_dist:{gps['ground_dist']:.0f}m")
            if "distance" in gps:
                parts.append(f"dist:{gps['distance']:.0f}m")
            if "speed" in gps:
                parts.append(f"spd:{gps['speed']:.1f}")
            if parts:
                gps_text2 = "  ".join(parts)
        scene["gps"] = gps  # store for report

        def add_markers(img):
            """Add markers: left = filename + scene/frame/time, right = GPS + telemetry (3 lines)."""
            # Left: filename (small), scene/frame/time (small) — shifted down
            draw_text_with_bg(img, video_fn, 10, 50, marker_font, 0.5, 1, (0, 255, 255))
            draw_text_with_bg(img, right_text_1, 10, 72, marker_font, 0.5, 1, (0, 255, 255))
            draw_text_with_bg(img, right_text_2, 10, 94, marker_font, 0.5, 1, (0, 255, 255))
            # Right: GPS (top, small), altitudes (small), yaw/pitch (small), fl/zoom (small)
            y_right = 50
            if gps_text:
                (tw, th), _ = cv2.getTextSize(gps_text, marker_font, 0.5, 1)
                draw_text_with_bg(img, gps_text, w - tw - 10, y_right, marker_font, 0.5, 1, (0, 255, 255))
                y_right += 22
            if gps_text3:
                (tw, th), _ = cv2.getTextSize(gps_text3, marker_font, 0.5, 1)
                draw_text_with_bg(img, gps_text3, w - tw - 10, y_right, marker_font, 0.5, 1, (0, 255, 255))
                y_right += 22
            # yaw/pitch line (small font)
            parts_yp = []
            if gps and "yaw" in gps:
                parts_yp.append(f"yaw:{gps['yaw']:.0f}")
            if gps and "pitch" in gps:
                parts_yp.append(f"pitch:{gps['pitch']:.0f}")
            if parts_yp:
                yp_text = ", ".join(parts_yp)
                (tw, th), _ = cv2.getTextSize(yp_text, marker_font, 0.5, 1)
                draw_text_with_bg(img, yp_text, w - tw - 10, y_right, marker_font, 0.5, 1, (0, 255, 255))
                y_right += 22
            # fl/zoom/gnd_dist line (small font)
            parts_fz = []
            if gps and "focal_len" in gps:
                parts_fz.append(f"fl:{gps['focal_len']:.0f}mm")
            if gps and "dzoom_ratio" in gps:
                parts_fz.append(f"zoom:{gps['dzoom_ratio']:.1f}x")
            if gps and "distance" in gps:
                parts_fz.append(f"dist:{gps['distance']:.0f}m")
            if gps and "speed" in gps:
                parts_fz.append(f"spd:{gps['speed']:.1f}")
            if parts_fz:
                fz_text = ", ".join(parts_fz)
                (tw, th), _ = cv2.getTextSize(fz_text, marker_font, 0.5, 1)
                draw_text_with_bg(img, fz_text, w - tw - 10, y_right, marker_font, 0.5, 1, (0, 255, 255))
            return img

        marked_orig = add_markers(frame.copy())
        save_with_exif(marked_orig, orig_path, gps, si, fi, video_fn)
        # V1: high contrast + grid (best for bright areas, color spotting)
        enhanced_v1 = enhance_frame_v1(frame)
        gridded_v1 = draw_grid(enhanced_v1, zones)
        add_markers(gridded_v1)
        save_with_exif(gridded_v1, grid_v1_path, gps, si, fi, video_fn)
        # V2: gentle CLAHE shadow recovery + grid (kept for reference, not sent to LLM)
        enhanced_v2 = enhance_frame_v2(frame)
        gridded_v2 = draw_grid(enhanced_v2, zones)
        add_markers(gridded_v2)
        save_with_exif(gridded_v2, grid_v2_path, gps, si, fi, video_fn)
        # V3: aggressive shadow recovery + grid (best for dark areas, caves, shadows)
        enhanced_v3 = enhance_frame_v3(frame)
        gridded_v3 = draw_grid(enhanced_v3, zones)
        add_markers(gridded_v3)
        save_with_exif(gridded_v3, grid_v3_path, gps, si, fi, video_fn)
        # V4: highlight recovery + grid (best for overexposed snow, blown highlights)
        enhanced_v4 = enhance_frame_v4(frame)
        gridded_v4 = draw_grid(enhanced_v4, zones)
        add_markers(gridded_v4)
        save_with_exif(gridded_v4, grid_v4_path, gps, si, fi, video_fn)

        scene["image_path"] = grid_v3_path  # default for LLM (aggressive shadow recovery)
        scene["image_path_v1"] = grid_v1_path  # high contrast variant (sent to LLM)
        scene["image_path_v2"] = grid_v2_path  # gentle CLAHE variant (reference only)
        scene["image_path_v3"] = grid_v3_path  # aggressive shadow variant (sent to LLM)
        scene["image_path_v4"] = grid_v4_path  # highlight recovery variant
        scene["orig_path"] = orig_path
        scene["enhanced_path"] = grid_v3_path

        print(f"  Scene {si:02d}: f{fi} t={scene['time_str']} "
              f"q={scene['best_score']:.2f} frames={scene['n_frames_in_scene']}")

    cap.release()

    # ── PASS 3.5: Color anomaly detection ─────────────────────────────
    if color_anomalies:
        print(f"\n{'='*60}")
        print("PASS 3.5: Color anomaly detection (scene-relative)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("PASS 3.5: Color anomaly detection (disabled, use --color-anomalies to enable)")
        print(f"{'='*60}")

    anomalies_dir = os.path.join(scenes_dir, "anomalies")
    os.makedirs(anomalies_dir, exist_ok=True)

    dynamics_results = {}
    t_dyn = time.time()
    if color_anomalies:
        for si, scene in enumerate(scenes):
            if si not in analyze_set:
                continue
            orig_path = scene.get("orig_path", "")
            if not orig_path or not os.path.exists(orig_path):
                continue

            img = cv2.imread(orig_path)
            if img is None:
                continue

            findings = detect_color_anomalies(img)

            base = f"scene_{si:02d}_f{scene['best_frame']:05d}"

             # Save annotated image to anomalies folder (only when findings exist)
            dyn_path = ""
            if findings:
                result_img = draw_color_findings(img, findings)
                 # Add findings count below existing text markers (same style)
                n_findings = len(findings)
                label = f"{n_findings} finding{'s' if n_findings != 1 else ''}"
                draw_text_with_bg(result_img, label, 10, 116, cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, 1, (0, 255, 255))
                dyn_path = os.path.join(anomalies_dir, f"{base}_color_anomalies.jpg")
                cv2.imwrite(dyn_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            scene["dynamics"] = build_dynamics(
                findings,
                image_file=os.path.basename(dyn_path) if findings else "",
            )
            dynamics_results[si] = scene["dynamics"]

            if findings:
                print(f"  Scene {si:04d} f{scene['best_frame']:05d}: "
                      f"{len(findings)} findings, zones={scene['dynamics']['flagged_zones']}")
                for f in findings:
                    print(f"    → {f['color']} Z{f['zone']} area={f['area']} "
                          f"conf={f['confidence']} bbox={f['bbox']}")

    print(f"  Color anomalies: {len(dynamics_results)} scenes, {time.time()-t_dyn:.1f}s")

    # ── PASS 4: LLM analysis (fast model on top scenes) ───────────────
    if llm_run:
        print(f"\n{'='*60}")
        print(f"PASS 4: LLM ANALYSIS ({fast_model}) — {len(analyze_set)} scenes")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"PASS 4: LLM ANALYSIS — SKIPPED (LLM off by default, use --llm-run)")
        print(f"{'='*60}")

    llm_error_count = 0
    LLM_MAX_ERRORS = 3  # Bail out after 3 consecutive errors

    if not llm_run:
        print(f"\n  LLM analysis skipped (off by default). Images and previews saved.")
    else:
        # Collect scenes to analyze (minimum quality filter for LLM)
        fast_scenes = [(si, scene) for si, scene in enumerate(scenes)
                       if si in analyze_set and "image_path" in scene
                       and scene.get("best_score", 0) >= LLM_MIN_QUALITY]
        skipped_low_q = [si for si, scene in enumerate(scenes)
                         if si in analyze_set and "image_path" in scene
                         and scene.get("best_score", 0) < LLM_MIN_QUALITY]
        if skipped_low_q:
            print(f"\n  ⚠ {len(skipped_low_q)} scenes skipped for LLM (quality < {LLM_MIN_QUALITY}): "
                  f"{['S%02d(q=%.2f)' % (si, scenes[si]['best_score']) for si in skipped_low_q]}")
            print(f"  These scenes are saved as images but not sent to LLM to save budget.")

        if llm_parallel > 0 and len(fast_scenes) > 1:
            # ── Parallel fast pass ────────────────────────────────────
            print(f"\n  Parallel mode: {llm_parallel} workers, {len(fast_scenes)} scenes")

            # Build task list: one task per (scene, variant)
            task_list = []
            task_meta = []  # (scene_idx, vlabel, vpath)
            for si, scene in fast_scenes:
                variants = get_llm_variants(scene, full=llm_pipeline_full)
                for vlabel, vpath in variants:
                      # Fast pass is ALWAYS single-stage (two-stage is deep-pass only).
                    task_list.append(lambda vpath=vpath: llm_analyze(
                        fast_model, vpath, prompt=sar_prompt))
                    task_meta.append((si, vlabel, vpath))

            t_fast_start = time.time()
            results = parallel_llm_batch(task_list, workers=llm_parallel)
            t_fast_total = time.time() - t_fast_start

            # Merge results back into scenes
            scene_variants = {}  # si -> {vlabel: (result, elapsed, err)}
            for idx, (si, vlabel, vpath) in enumerate(task_meta):
                if si not in scene_variants:
                    scene_variants[si] = []
                scene_variants[si].append((vlabel, vpath, results[idx]))

            for si, scene in fast_scenes:
                variant_data = scene_variants.get(si, [])
                merged_findings = []
                merged_found = False
                merged_conf = 0
                merged_terrain = ""
                variant_results = {}

                for vlabel, vpath, (result, elapsed, err) in variant_data:
                    if err:
                        print(f"  [{si:3d}] {vlabel} ERROR: {err[:150]}")
                        variant_results[vlabel] = {"error": err[:500]}
                        continue
                    variant_results[vlabel] = result
                    v_found = result.get("objects_found", False)
                    v_conf = result.get("confidence", 0)
                    v_findings = result.get("findings", [])
                    print(f"  [{si:3d}] f{scene['best_frame']:5d} {vlabel}: "
                          f"found={v_found} conf={v_conf} "
                          f"findings={len(v_findings)} ({elapsed:.1f}s)")
                    if v_found:
                        merged_found = True
                        merged_conf = max(merged_conf, v_conf)
                        for fnd in v_findings:
                            fnd["source_variant"] = vlabel
                            merged_findings.append(fnd)
                            print(f"         → {fnd.get('type','?')} Z{fnd.get('zone','?')} "
                                  f"color={fnd.get('color','?')} "
                                  f"conf={fnd.get('confidence',0)} "
                                  f"[{vlabel}] "
                                  f"{fnd.get('description','')[:60]}")
                    merged_terrain = result.get("terrain", merged_terrain)

                scene["llm_fast"] = {
                    "objects_found": merged_found,
                    "confidence": merged_conf,
                    "description": variant_results.get("v3_aggressive_shadow",
                                      variant_results.get("v1_high_contrast", {})).get("description", ""),
                    "findings": merged_findings,
                    "terrain": merged_terrain,
                    "variant_results": variant_results,
                }
                scene["llm_fast"].update(
                    {k: v for k, v in variant_results.get("v3_aggressive_shadow",
                        variant_results.get("v1_high_contrast", {})).items()
                     if k not in ("objects_found", "confidence", "description",
                                  "findings", "terrain", "llm_time", "llm_model")}
                )
                scene["llm_fast"]["llm_time"] = sum(
                    v.get("llm_time", 0) for v in variant_results.values()
                    if isinstance(v, dict))
                scene["llm_fast"]["llm_model"] = fast_model

            print(f"\n  Fast pass done: {len(fast_scenes)} scenes in {t_fast_total:.0f}s")

        else:
            # ── Sequential fast pass (original) ────────────────────────
            for si, scene in fast_scenes:
                # Graceful bail-out if LLM is unavailable
                if llm_error_count >= LLM_MAX_ERRORS:
                    print(f"\n  ⚠ LLM unavailable ({llm_error_count} consecutive errors) — skipping remaining LLM analysis")
                    print(f"  Images saved successfully. Re-run when LLM is available.")
                    break

                print(f"\n  [{si+1}/{len(scenes)}] Scene {si:02d} "
                      f"f{scene['best_frame']} t={scene['time_str']}...")

                # Analyze variants (v3 only by default, all variants with --llm-pipeline-full)
                variants = get_llm_variants(scene, full=llm_pipeline_full)

                merged_findings = []
                merged_found = False
                merged_conf = 0
                merged_terrain = ""
                variant_results = {}
                scene_had_error = False

                for vlabel, vpath in variants:
                    result, elapsed, err = llm_analyze(fast_model, vpath,
                                                        prompt=sar_prompt)
                    if err:
                        print(f"    {vlabel} ERROR: {err[:150]}")
                        variant_results[vlabel] = {"error": err[:500]}
                        scene_had_error = True
                        continue
                    llm_error_count = 0  # Reset on success
                    variant_results[vlabel] = result
                    v_found = result.get("objects_found", False)
                    v_conf = result.get("confidence", 0)
                    v_findings = result.get("findings", [])
                    print(f"    {vlabel}: found={v_found} conf={v_conf} "
                          f"findings={len(v_findings)} ({elapsed:.1f}s)")
                    if v_found:
                        merged_found = True
                        merged_conf = max(merged_conf, v_conf)
                        for fnd in v_findings:
                            fnd["source_variant"] = vlabel
                            merged_findings.append(fnd)
                            print(f"      → {fnd.get('type','?')} Z{fnd.get('zone','?')} "
                                  f"color={fnd.get('color','?')} "
                                  f"conf={fnd.get('confidence',0)} "
                                  f"[{vlabel}] "
                                  f"{fnd.get('description','')[:60]}")
                        # Stop-on-first-find: skip remaining variants (not for --llm-pipeline-full)
                        if not llm_pipeline_full:
                            remaining = len(variants) - (variants.index((vlabel, vpath)) + 1)
                            if remaining > 0:
                                print(f"    [stop-on-find] {remaining} remaining variant(s) skipped")
                            break
                    merged_terrain = result.get("terrain", merged_terrain)

                # Merge into single result
                scene["llm_fast"] = {
                    "objects_found": merged_found,
                    "confidence": merged_conf,
                    "description": variant_results.get("v3_aggressive_shadow",
                                      variant_results.get("v1_high_contrast", {})).get("description", ""),
                    "findings": merged_findings,
                    "terrain": merged_terrain,
                    "variant_results": variant_results,
                }
                scene["llm_fast"].update(
                    {k: v for k, v in variant_results.get("v3_aggressive_shadow",
                        variant_results.get("v1_high_contrast", {})).items()
                     if k not in ("objects_found", "confidence", "description",
                                  "findings", "terrain", "llm_time", "llm_model")}
                )
                scene["llm_fast"]["llm_time"] = sum(
                    v.get("llm_time", 0) for v in variant_results.values()
                    if isinstance(v, dict))
                scene["llm_fast"]["llm_model"] = fast_model
                if scene_had_error:
                    llm_error_count += 1

    # ── PASS 5: Deep LLM on positives/uncertain ───────────────────────
    # Heuristic scorer — ranks scenes by deep LLM value
    TARGET_COLORS = {"orange", "bright orange", "reddish-orange", "blue", "dark blue",
                     "blue-grey", "black", "dark grey", "dark gray", "dark brown"}

    def deep_priority_score(scene):
        """Heuristic score for deep LLM budget allocation.

        Factors:
        - Finding count (more findings = higher priority)
        - Target jacket colors (orange/blue/dark) boost
        - Fast confidence
        - Color anomaly zone overlap with LLM findings
        - Cross-scene persistence (same zone+color in adjacent scenes)
        """
        score = 0
        llm = scene.get("llm_fast", {})
        if not isinstance(llm, dict):
            return 0
        findings = llm.get("findings", [])
        score += len(findings) * 2

        llm_zones = set()
        for f in findings:
            color = (f.get("color", "") or "").lower()
            if any(tc in color for tc in TARGET_COLORS):
                score += 5
            score += f.get("confidence", 0) * 3
            zone = f.get("zone", "")
            if zone:
                llm_zones.add(zone)

        # Color anomaly overlap
        dynamics = scene.get("dynamics", {})
        if isinstance(dynamics, dict):
            ca_findings = dynamics.get("color_findings", [])
            if ca_findings:
                ca_zones = {f"Z{f.get('zone', '')}" for f in ca_findings}
                overlap = llm_zones & ca_zones
                score += len(overlap) * 4

        return score

    positive_scenes = [s for s in scenes
                        if s.get("llm_fast", {}).get("objects_found", False)
                        or s.get("llm_fast", {}).get("confidence", 0) > 0.4]
    # Chancepeek: also include scenes where fast found nothing
    if llm_pipeline_chancepeek:
        negative_scenes = [s for s in scenes
                           if not s.get("llm_fast", {}).get("objects_found", False)
                           and s.get("llm_fast", {}).get("confidence", 0) <= 0.4]
        candidate_pool = positive_scenes + negative_scenes
        peek_tag = f" (+{len(negative_scenes)} chancepeek)" if negative_scenes else ""

        # ── Deep budget warnings ────────────────────────────────────
        if deep_top_n < 10:
            print(f"  ⚠ Chancepeek with low deep budget ({deep_top_n}) — most fast-negative scenes won't be checked.")
            print(f"    Consider --llm-deep-max-scenes 15+.")
        total_check = len(positive_scenes) + len(negative_scenes)
        if total_check > deep_top_n:
            pct = round(deep_top_n / total_check * 100)
            print(f"  ⚠ Deep budget ({deep_top_n}) covers only {pct}% of {total_check} scenes with chancepeek enabled.")
    else:
        candidate_pool = positive_scenes
        peek_tag = ""

        # ── Deep budget warnings ────────────────────────────────────
        if len(positive_scenes) > deep_top_n:
            print(f"  ⚠ More fast positives ({len(positive_scenes)}) than deep budget ({deep_top_n}) — some positives won't get deep confirmation.")
            print(f"    Consider --llm-deep-max-scenes {len(positive_scenes)}+.")

    # ── Rank candidates by heuristic priority score ────────────────
    if len(candidate_pool) > deep_top_n:
        scored = [(deep_priority_score(s), s) for s in candidate_pool]
        scored.sort(key=lambda x: -x[0])
        deep_targets = [s for _, s in scored[:deep_top_n]]
        # Re-sort chronologically for output
        deep_targets.sort(key=lambda s: s.get("best_frame", 0))
        print(f"\n  Deep budget triage: {len(candidate_pool)} candidates → top {len(deep_targets)} by priority score")
        print(f"  Top 5 scores: {[(s.get('best_frame', 0), round(sc, 1)) for sc, s in scored[:5]]}")
        print(f"  Cut-off score: {round(scored[deep_top_n-1][0], 1)} (scene f{scored[deep_top_n-1][1].get('best_frame', 0)})")
    else:
        deep_targets = candidate_pool

    if llm_run and deep_targets and deep_model:
        stage_tag = " [TWO-STAGE]" if llm_two_stage else ""
        par_tag = f" [{llm_parallel} workers]" if llm_parallel > 0 else ""
        print(f"\n{'='*60}")
        print(f"PASS 5: DEEP LLM ANALYSIS ({deep_model}){stage_tag}{peek_tag}{par_tag}")
        if llm_two_stage:
            print(f"  Vision: {deep_model} → Reasoning: {llm_reasoning_model}")
        print(f"{'='*60}")

        if llm_parallel > 0 and len(deep_targets) > 1:
            # ── Parallel deep pass ─────────────────────────────────────
            task_list = []
            task_meta = []  # (scene_idx, vlabel, vpath)
            for scene in deep_targets:
                variants = []
                if scene.get("image_path_v3"):
                    variants.append(("v3_aggressive_shadow", scene["image_path_v3"]))
                if not variants:
                    variants.append(("default", scene["image_path"]))
                for vlabel, vpath in variants:
                    if llm_two_stage:
                        task_list.append(lambda vpath=vpath: two_stage_analyze(
                            deep_model, llm_reasoning_model, vpath,
                            mission_context=mission_context, timeout_vision=180))
                    else:
                        task_list.append(lambda vpath=vpath: llm_analyze(
                            deep_model, vpath, timeout=180, prompt=sar_prompt))
                    task_meta.append((scene, vlabel, vpath))

            t_deep_start = time.time()
            results = parallel_llm_batch(task_list, workers=llm_parallel)
            t_deep_total = time.time() - t_deep_start

            # Merge results back into scenes
            scene_results = {}  # id(scene) -> (scene, [(vlabel, vpath, result_tuple)])
            for idx, (scene, vlabel, vpath) in enumerate(task_meta):
                sid = id(scene)
                if sid not in scene_results:
                    scene_results[sid] = (scene, [])
                scene_results[sid][1].append((vlabel, vpath, results[idx]))

            for sid, (scene, variant_data) in scene_results.items():
                merged_findings = []
                merged_found = False
                merged_conf = 0
                variant_results = {}

                for vlabel, vpath, (result, elapsed, err) in variant_data:
                    if err:
                        print(f"  f{scene['best_frame']:5d} {vlabel} ERROR: {err[:150]}")
                        variant_results[vlabel] = {"error": err[:500]}
                        continue
                    variant_results[vlabel] = result
                    v_found = result.get("objects_found", False)
                    v_conf = result.get("confidence", 0)
                    v_findings = result.get("findings", [])
                    st_tag = " [2-stage]" if llm_two_stage else ""
                    print(f"  f{scene['best_frame']:5d} {vlabel}: "
                          f"found={v_found} conf={v_conf} ({elapsed:.1f}s){st_tag}")
                    if v_found:
                        merged_found = True
                        merged_conf = max(merged_conf, v_conf)
                        for fnd in v_findings:
                            fnd["source_variant"] = vlabel
                            merged_findings.append(fnd)
                            print(f"         → {fnd.get('type','?')} Z{fnd.get('zone','?')} "
                                  f"color={fnd.get('color','?')} "
                                  f"conf={fnd.get('confidence',0)} "
                                  f"[{vlabel}] "
                                  f"{fnd.get('description','')[:60]}")

                scene["llm_deep"] = {
                    "objects_found": merged_found,
                    "confidence": merged_conf,
                    "findings": merged_findings,
                    "variant_results": variant_results,
                }
                best_v = variant_results.get("v3_aggressive_shadow",
                             variant_results.get("v1_high_contrast", {}))
                if isinstance(best_v, dict):
                    scene["llm_deep"]["description"] = best_v.get("description", "")
                    scene["llm_deep"]["terrain"] = best_v.get("terrain", "")

            print(f"\n  Deep pass done: {len(deep_targets)} scenes in {t_deep_total:.0f}s")

        else:
            # ── Sequential deep pass (original) ────────────────────────
            for si, scene in enumerate(deep_targets):
                print(f"\n  [{si+1}/{len(deep_targets)}] Scene "
                      f"f{scene['best_frame']} t={scene['time_str']}...")

                # Analyze v3 only with deep model
                variants = []
                if scene.get("image_path_v3"):
                    variants.append(("v3_aggressive_shadow", scene["image_path_v3"]))
                if not variants:
                    variants.append(("default", scene["image_path"]))

                merged_findings = []
                merged_found = False
                merged_conf = 0
                variant_results = {}

                for vlabel, vpath in variants:
                    if llm_two_stage:
                        result, elapsed, err = two_stage_analyze(
                            deep_model, llm_reasoning_model, vpath,
                            mission_context=mission_context, timeout_vision=180)
                    else:
                        result, elapsed, err = llm_analyze(deep_model, vpath,
                                                         timeout=180, prompt=sar_prompt)
                    if err:
                        print(f"    {vlabel} ERROR: {err[:150]}")
                        variant_results[vlabel] = {"error": err[:500]}
                        continue
                    variant_results[vlabel] = result
                    v_found = result.get("objects_found", False)
                    v_conf = result.get("confidence", 0)
                    v_findings = result.get("findings", [])
                    stage_tag = " [2-stage]" if llm_two_stage else ""
                    print(f"    {vlabel}: found={v_found} conf={v_conf} ({elapsed:.1f}s){stage_tag}")
                    if v_found:
                        merged_found = True
                        merged_conf = max(merged_conf, v_conf)
                        for fnd in v_findings:
                            fnd["source_variant"] = vlabel
                            merged_findings.append(fnd)
                            print(f"      → {fnd.get('type','?')} Z{fnd.get('zone','?')} "
                                  f"color={fnd.get('color','?')} "
                                  f"conf={fnd.get('confidence',0)} "
                                  f"[{vlabel}] "
                                  f"{fnd.get('description','')[:60]}")

                scene["llm_deep"] = {
                    "objects_found": merged_found,
                    "confidence": merged_conf,
                    "findings": merged_findings,
                    "variant_results": variant_results,
                }
                # Copy description from best variant
                best_v = variant_results.get("v3_aggressive_shadow",
                             variant_results.get("v1_high_contrast", {}))
                if isinstance(best_v, dict):
                    scene["llm_deep"]["description"] = best_v.get("description", "")
                    scene["llm_deep"]["terrain"] = best_v.get("terrain", "")

    # ── Contact sheet (shared builder from previews.py) ────────────────
    build_contact_sheet(scenes, video_path, out_dir, fps=fps,
                        enhance_fn=enhance_frame,
                        grid_fn=lambda f: draw_grid(f, zones))

    # ── LLM findings preview (shared builder from previews.py) ──────────
    preview_report = {"scenes": scenes, "video_info": {"fps": fps}}
    build_llm_findings_preview(out_dir, report=preview_report, scenes_dir=scenes_dir)

    # ── Report ─────────────────────────────────────────────────────────
    # Clean scenes for JSON (remove numpy arrays, raw frames)
    clean_scenes = []
    for si, scene in enumerate(scenes):
        cs = {
            "scene_id": si,
            "best_frame": scene["best_frame"],
            "timestamp": scene.get("timestamp", 0),
            "time_str": scene.get("time_str", ""),
            "quality_score": round(scene["best_score"], 3),
            "n_frames_in_scene": scene["n_frames_in_scene"],
            "frame_range": scene["frame_range"],
            "image_file": os.path.basename(scene.get("image_path", "")),
            "image_file_v1": os.path.basename(scene.get("image_path_v1", "")),
            "image_file_v2": os.path.basename(scene.get("image_path_v2", "")),
            "image_file_v3": os.path.basename(scene.get("image_path_v3", "")),
            "image_file_v4": os.path.basename(scene.get("image_path_v4", "")),
        }
        if scene.get("gps"):
            cs["gps"] = scene["gps"]
        if "llm_fast" in scene:
            cs["llm_fast"] = scene["llm_fast"]
        if "llm_deep" in scene:
            cs["llm_deep"] = scene["llm_deep"]
        if "dynamics" in scene:
            cs["dynamics"] = scene["dynamics"]
        clean_scenes.append(cs)

    analysis_end = time.time()
    report = {
        "video": video_path,
        "recording_time": recording_time,
        "analysis_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(analysis_start)),
        "analysis_end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(analysis_end)),
        "analysis_elapsed_sec": round(analysis_end - analysis_start, 1),
        "video_info": {
            "width": w, "height": h, "fps": round(fps, 2),
            "total_frames": total, "duration_sec": round(total / fps, 1)
        },
        "analysis_params": {
            "stride": stride_mode,
            "quality_thresh": quality_thresh,
            "scene_sim_thresh": scene_sim_thresh,
            "fast_model": fast_model,
            "deep_model": deep_model,
            "mission_context": mission_context.to_dict() if mission_context else None,
        },
        "total_frames_scanned": len(quality_frames),
        "quality_frames": len(quality_frames),
        "total_scenes": len(scenes),
        "llm_positives": len([s for s in scenes if s.get("llm_fast", {}).get("objects_found", False)]),
        "deep_llm_confirmed": len([s for s in scenes if s.get("llm_deep", {}).get("objects_found", False)]),
        "findings_total": sum(len(s.get("llm_fast", {}).get("findings", [])) for s in scenes
                              if isinstance(s.get("llm_fast"), dict)) +
                          sum(len(s.get("llm_deep", {}).get("findings", [])) for s in scenes
                              if isinstance(s.get("llm_deep"), dict)),
        "scenes": clean_scenes,
    }

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # ── Build preview videos (hq + lq + color anomalies) ─────────────
    if build_preview:
        preview_report = {"scenes": scenes, "video_info": {"fps": fps}}
        build_hq_lq_previews(out_dir, report=preview_report, scenes_dir=scenes_dir)

        if color_anomalies:
            build_color_anomalies_preview(out_dir, scenes_dir, report=preview_report)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Frames scanned: {len(quality_frames)} sampled")
    print(f"Quality frames: {len(quality_frames)}")
    print(f"Distinct scenes: {len(scenes)}")

    positives = [s for s in scenes
                  if s.get("llm_fast", {}).get("objects_found", False)]
    print(f"\nLLM POSITIVES: {len(positives)}")
    for s in positives:
        llm = s.get("llm_fast", {})
        print(f"  Scene f{s['best_frame']} t={s.get('time_str','')} "
              f"conf={llm.get('confidence',0)}")
        for fnd in llm.get("findings", []):
            print(f"    → {fnd.get('type','?')} {fnd.get('zone','')} "
                  f"color={fnd.get('color','?')} {fnd.get('description','')[:100]}")

    deep_positives = [s for s in scenes
                       if s.get("llm_deep", {}).get("objects_found", False)]
    if deep_positives:
        print(f"\nDEEP LLM CONFIRMED: {len(deep_positives)}")
        for s in deep_positives:
            llm = s.get("llm_deep", {})
            print(f"  Scene f{s['best_frame']} t={s.get('time_str','')} "
                  f"conf={llm.get('confidence',0)}")
            for fnd in llm.get("findings", []):
                print(f"    → {fnd.get('type','?')} {fnd.get('zone','')} "
                      f"color={fnd.get('color','?')} {fnd.get('description','')[:100]}")

    print(f"\nReport: {report_path}")
    print(f"Scenes: {scenes_dir}")
    print(f"Contact sheet: {out_dir}/contact_sheet.jpg")

    return report


def main():
    p = argparse.ArgumentParser(description="Alpine Zoom — SAR Scene Analyzer")
    p.add_argument("video", help="Path to drone video")
    p.add_argument("-o", "--output", default=None,
                   help="Output dir (default: <video_basename>.AZ/, e.g. video1.mp4.AZ/)")
    p.add_argument("--stride", default="dynamic",
                   help="Frame sampling interval. Integer (every N frames) or 'dynamic' (motion-adaptive, default). "
                        "Dynamic: dense sampling during fast motion/zoom, sparse during slow pans.")
    p.add_argument("--quality", type=float, default=0.5,
                   help="Minimum quality threshold (0-1)")
    p.add_argument("--scene-sim", type=float, default=0.82,
                   help="Scene similarity threshold (0-1, higher=more scenes)")
    p.add_argument("--llm-fast-model", default="gemma4:31b-cloud")
    p.add_argument("--llm-deep-model", default="qwen3.5:397b-cloud")
    p.add_argument("--llm-deep-max-scenes", type=int, default=20,
                   help="Max scenes for deep LLM (positives, or positives+chancepeek negatives)")
    p.add_argument("--llm-scenes-cap", default=50,
                   help="Max scenes to send to fast LLM (spread across timeline). Integer (default 50) or 'all'.")
    p.add_argument("--helicopter", action="store_true",
                   help="Helicopter footage mode — LLM ignores interior passengers/elements")
    p.add_argument("--llm-pipeline", default=None,
                   choices=["fast", "chancepeek", "max"],
                   help="LLM pipeline mode: 'fast' (default, v3 only, stop-on-find, deep on positives), "
                        "'chancepeek' (deep also runs on fast negatives), "
                        "'max' (all variants, no stop-on-find, deep on all variants for positives)")
    p.add_argument("--from", type=float, default=None, dest="from_sec",
                   help="Start processing from this time (seconds). Default: beginning")
    p.add_argument("--to", type=float, default=None, dest="to_sec",
                   help="Stop processing at this time (seconds). Default: end")
    p.add_argument("--llm-run", action="store_true",
                   help="Run LLM analysis (off by default — generates scenes/images only)")
    p.add_argument("--build-preview", action="store_true",
                   help="Build preview videos (hq/lq/color anomalies). Off by default.")
    p.add_argument("--dedup-thresh", type=float, default=DEDUP_COMBINED_THRESH,
                   help="Scene deduplication threshold (0.90 default, 0 to disable). "
                        "Merges near-duplicate scenes across time gaps.")
    p.add_argument("--color-anomalies", action="store_true",
                   help="Enable variant dynamics / color anomaly detection (disabled by default)")
    p.add_argument("--recording-time", default=None, dest="recording_time_override",
                   help="Override recording time (ISO format, e.g. '2026-08-15 12:08:14'). "
                        "Use when camera clock is wrong. Used for abs_time markers and report.")
    p.add_argument("--context-file", default=None, dest="context_file",
                   help="Path to JSON mission context config. Overrides --context-preset and --helicopter.")
    p.add_argument("--context-preset", default=None, dest="context_preset",
                       help="Named preset: 'sar', 'sar-heli'. "
                         "Overrides --helicopter. Default (none specified) loads 'sar'.")
    p.add_argument("--llm-no-two-stage", action="store_true", dest="llm_no_two_stage",
                   help="Disable two-stage LLM mode (on by default). Use single-stage: one LLM call per image. "
                        "Two-stage: vision model describes scene, reasoning model concludes. "
                        "Uses --llm-fast-model as vision, --llm-reasoning-model as reasoning.")
    p.add_argument("--llm-reasoning-model", default="glm-5.1:cloud", dest="llm_reasoning_model",
                   help="Reasoning model for two-stage mode (default: glm-5.1:cloud). "
                        "Text-only — no image processing.")
    p.add_argument("--llm-parallel", type=int, default=None, dest="llm_parallel",
                   help="Number of parallel LLM workers (default 0 = sequential). "
                        "Cloud models can handle 4+ concurrent requests (~5x speedup). "
                        "Local models should stay at 0 (VRAM-bound).")
    p.add_argument("--run-standard", action="store_true",
                   help="Preset: full analysis with LLM. Sets --color-anomalies --build-preview "
                        "--llm-run --llm-parallel 4 --llm-pipeline chancepeek. "
                        "Individual flags override the preset.")
    p.add_argument("--run-light", action="store_true",
                   help="Preset: scenes + images + color anomalies + previews, no LLM. "
                        "Sets --color-anomalies --build-preview. "
                        "Individual flags override the preset.")
    args = p.parse_args()

    # Apply presets — explicit flags override preset defaults.
    if args.run_standard:
        args.color_anomalies = True
        args.build_preview = True
        args.llm_run = True
        if args.llm_parallel is None:
            args.llm_parallel = 4
        if args.llm_pipeline is None:
            args.llm_pipeline = "chancepeek"
    if args.run_light:
        args.color_anomalies = True
        args.build_preview = True

    # Apply final defaults (after presets so presets can set them)
    if args.llm_parallel is None:
        args.llm_parallel = 0
    if args.llm_pipeline is None:
        args.llm_pipeline = "fast"

     # Default output dir: video filename + ".AZ" (e.g. video1.mp4.AZ/).
      # The .AZ suffix avoids colliding with the source video file when
      # running from the video's own directory.
    if args.output is None:
        args.output = os.path.basename(args.video) + ".AZ"

     # Resolve mission context: --context-file > --context-preset > --helicopter > default (sar)
    mission_context = get_context(
        preset=args.context_preset,
        context_file=args.context_file,
        helicopter=args.helicopter,
    )

    analyze_video(
        args.video, args.output, args.quality, args.scene_sim, args.stride,
        args.llm_fast_model, args.llm_deep_model, args.llm_deep_max_scenes, args.llm_scenes_cap,
        helicopter=args.helicopter,
        llm_pipeline=args.llm_pipeline,
        stride_mode=args.stride,
        from_sec=args.from_sec,
        to_sec=args.to_sec,
        recording_time_override=args.recording_time_override,
        color_anomalies=args.color_anomalies,
        llm_run=args.llm_run,
        build_preview=args.build_preview,
        dedup_thresh=args.dedup_thresh,
        mission_context=mission_context,
        llm_two_stage=not args.llm_no_two_stage,
        llm_reasoning_model=args.llm_reasoning_model,
        llm_parallel=args.llm_parallel,
     )


if __name__ == "__main__":
    main()
