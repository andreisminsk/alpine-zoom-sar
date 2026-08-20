"""Preview video building for SAR video analysis.

All preview video builders (cli/previews.py, cli/llm_preview.py,
and the inline builders in alpine_zoom.video) import from here.

Extracted from alpine_zoom.video (source of truth).
"""
import sys
import os
import re
import json
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.common import (
    scene_sort_key, parse_scene_frame, draw_text_with_bg,
    extract_zone_findings, build_scene_findings_map, draw_zone_highlights,
    find_scene_image, find_all_variants, SCENE_VARIANTS,
)

# ── Constants ─────────────────────────────────────────────────────────

PREVIEW_FPS = 10
PREVIEW_DURATION_HQ_LQ = 0.5   # seconds per image for hq/lq previews
PREVIEW_DURATION_FINDINGS = 2.0  # seconds per image for llm_findings preview


# ── Core video builder ────────────────────────────────────────────────

def build_preview_video(images, out_path, fps=30.0, duration_sec=PREVIEW_DURATION_HQ_LQ,
                         zone_findings_map=None, findings_text_fn=None, label="preview"):
    """Build a preview video from a list of images.

    Args:
        images: list of image paths (sorted chronologically by scene_sort_key)
        out_path: output .mp4 path
        fps: source video fps (for display purposes)
        duration_sec: seconds to show each image
        zone_findings_map: {scene_num: {zone_num: max_conf}} for zone highlighting
        findings_text_fn: optional fn(scene_dict, img_path) -> list of text lines
                           to overlay below baked-in markers
        label: display label for progress messages
    """
    if not images:
        print(f"  No images for {label} — skipping")
        return

    images = sorted(images, key=scene_sort_key)

    sample = cv2.imread(images[0])
    if sample is None:
        print(f"  ERROR: cannot read {images[0]}")
        return
    h, w = sample.shape[:2]

    preview_fps = PREVIEW_FPS
    frames_per_img = int(preview_fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, preview_fps, (w, h))
    if not writer.isOpened():
        print(f"  ERROR: cannot open VideoWriter for {out_path}")
        return

    print(f"  Building {label}: {len(images)} images -> {out_path}")
    print(f"  Resolution: {w}x{h}, {preview_fps}fps, {duration_sec}s/image")

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"    SKIP (unreadable): {os.path.basename(img_path)}")
            continue

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        # Zone highlighting
        if zone_findings_map is not None:
            scene_num, _ = parse_scene_frame(img_path)
            frame = draw_zone_highlights(frame, zone_findings_map.get(scene_num, {}))

        # Findings text overlay
        if findings_text_fn is not None:
            scene_num, _ = parse_scene_frame(img_path)
            lines = findings_text_fn(scene_num, img_path)
            if lines:
                y = 120
                for line in lines:
                    draw_text_with_bg(frame, line, margin, y, font, 0.5, 1, (0, 255, 255))
                    y += 22
                    if y > h - 30:
                        break

        for _ in range(frames_per_img):
            writer.write(frame)

    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    total_sec = len(images) * duration_sec
    print(f"  Done: {size_mb:.1f} MB, {len(images)} images, {total_sec:.1f}s total")


# ── Scene image collection ────────────────────────────────────────────

def collect_scene_images(scenes_dir, subfolder, variants=None):
    """Collect all variant images for scenes in a subfolder (hq/ or lq/).

    Returns a chronologically sorted list of image paths.
    """
    if variants is None:
        variants = SCENE_VARIANTS

    d = os.path.join(scenes_dir, subfolder)
    if not os.path.isdir(d):
        return []

    # Find unique scene bases
    bases = set()
    for f in os.listdir(d):
        if f.endswith(".jpg"):
            base = f
            for v in variants:
                base = base.replace(v, "")
            bases.add(base)

    images = []
    for base in sorted(bases, key=lambda b: scene_sort_key(os.path.join(d, b + "_orig.jpg"))):
        for v in variants:
            p = os.path.join(d, base + v)
            if os.path.exists(p):
                images.append(p)
    return images


# ── HQ/LQ preview builder ─────────────────────────────────────────────

def build_hq_lq_previews(output_dir, report=None, scenes_dir=None):
    """Build preview_hq.mp4 and preview_lq.mp4 from scene images.

    Args:
        output_dir: video output directory (contains report.json + scenes/)
        report: pre-loaded report dict (optional — loaded from output_dir if None)
        scenes_dir: pre-resolved scenes dir (optional)
    """
    if scenes_dir is None:
        scenes_dir = os.path.join(output_dir, "scenes")

    if report is None:
        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        else:
            report = {}

    fps = report.get("video_info", {}).get("fps", 30.0)
    scenes = report.get("scenes", [])
    zone_findings_map = build_scene_findings_map(scenes) if scenes else {}

    hq_images = collect_scene_images(scenes_dir, "hq")
    lq_images = collect_scene_images(scenes_dir, "lq")

    print(f"Output dir: {output_dir}")
    print(f"fps: {fps:.1f}, HQ images: {len(hq_images)}, LQ images: {len(lq_images)}")
    print()

    build_preview_video(hq_images, os.path.join(output_dir, "preview_hq.mp4"),
                         fps=fps, duration_sec=PREVIEW_DURATION_HQ_LQ,
                         zone_findings_map=zone_findings_map, label="HQ")
    print()
    build_preview_video(lq_images, os.path.join(output_dir, "preview_lq.mp4"),
                         fps=fps, duration_sec=PREVIEW_DURATION_HQ_LQ,
                         zone_findings_map=zone_findings_map, label="LQ")


# ── Color anomalies preview ───────────────────────────────────────────

def build_color_anomalies_preview(output_dir, scenes_dir, report=None):
    """Build preview_color_anomalies.mp4 from anomaly images + their originals."""
    if report is None:
        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        else:
            report = {}

    anomalies_dir = os.path.join(scenes_dir, "anomalies")
    if not os.path.isdir(anomalies_dir):
        return

    print("Building color anomalies preview")
    ca_pairs = []
    scenes = report.get("scenes", [])

    def find_orig(base):
        """Find the original image for a scene base in hq/ or lq/.

        `base` may be either a bare scene base (e.g. "scene_05_f01101", needs
        "_orig.jpg" appended) or a full original filename already including
        "_orig" (e.g. "scene_05_f01101_orig", from standalone az-colors output).
        """
        candidates = [base + "_orig.jpg", base + ".jpg"]
        for cand in candidates:
            for sub in ("hq", "lq", ""):
                p = os.path.join(scenes_dir, sub, cand) if sub else os.path.join(scenes_dir, cand)
                if os.path.exists(p):
                    return p
        return None

    # Primary path: report-driven (dynamics.color_findings populated by main pipeline)
    for s in scenes:
        if not s.get("dynamics", {}).get("color_findings"):
            continue
        si = s.get("scene_id", 0)
        fi = s.get("best_frame", 0)
        base = f"scene_{si:02d}_f{fi:05d}"
        # Try both naming conventions: pipeline writes "<base>_color_anomalies.jpg",
        # standalone az-colors writes "<base>_orig_color_anomalies.jpg".
        anomaly_path = None
        for cand in (base + "_color_anomalies.jpg", base + "_orig_color_anomalies.jpg"):
            p = os.path.join(anomalies_dir, cand)
            if os.path.exists(p):
                anomaly_path = p
                break
        orig_path = find_orig(base)
        if orig_path and anomaly_path:
            ca_pairs.append((orig_path, anomaly_path))

    # Fallback: report has no dynamics (e.g. anomalies generated by standalone az-colors
    # after report.json was written). Scan the anomalies/ directory directly.
    if not ca_pairs:
        print("  No dynamics in report.json — scanning anomalies/ directory directly")
        for fn in sorted(os.listdir(anomalies_dir)):
            if not fn.endswith("_color_anomalies.jpg"):
                continue
            anomaly_path = os.path.join(anomalies_dir, fn)
            # Strip the anomaly suffix to recover the original base.
            base = fn[:-len("_color_anomalies.jpg")]
            orig_path = find_orig(base)
            if orig_path:
                ca_pairs.append((orig_path, anomaly_path))

    if not ca_pairs:
        print("  No color anomaly images found — skipping")
        return

    # Interleave orig + anomaly
    images = []
    for orig, anomaly in ca_pairs:
        images.extend([orig, anomaly])

    build_preview_video(images, os.path.join(output_dir, "preview_color_anomalies.mp4"),
                         duration_sec=1.0, label="color_anomalies")


# ── LLM findings preview ──────────────────────────────────────────────

def copy_llm_finding_scenes(scenes_dir, report):
    """Copy scenes with LLM findings to llm_findings/ folder.

    Returns list of (scene_dict, image_path) for preview building.
    """
    findings_dir = os.path.join(scenes_dir, "llm_findings")
    os.makedirs(findings_dir, exist_ok=True)

    scenes = report.get("scenes", [])
    findings_scenes = []
    preview_images = []

    for s in scenes:
        llm_fast = s.get("llm_fast", {})
        llm_deep = s.get("llm_deep", {})
        has = False
        if isinstance(llm_fast, dict) and llm_fast.get("objects_found", False):
            has = True
        if isinstance(llm_deep, dict) and llm_deep.get("objects_found", False):
            has = True
        if not has:
            continue

        si = s.get("scene_id", 0)
        fi = s.get("best_frame", 0)
        base = f"scene_{si:02d}_f{fi:05d}"

        # Copy ALL variants (orig + v1..v4) that exist, matching az-video's inline
        # behavior. Search hq/ first, then lq/, then scenes/ root.
        for v in SCENE_VARIANTS:
            fname = base + v
            for sub in ("hq", "lq", ""):
                src = os.path.join(scenes_dir, sub, fname) if sub else os.path.join(scenes_dir, fname)
                if os.path.exists(src):
                    dst = os.path.join(findings_dir, os.path.basename(src))
                    img = cv2.imread(src)
                    if img is not None:
                        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        preview_images.append((s, dst))
                    break   # found this variant in one subdir; move to next variant

        findings_scenes.append(s)
        print(f"  Scene {si:04d} f{fi} t={s.get('time_str', '')}")

    print(f"  {len(findings_scenes)} scenes with findings copied to llm_findings/")
    return findings_scenes, preview_images


def _findings_text_for_scene(scene_dict, img_path):
    """Build text lines for LLM findings overlay."""
    lines = []
    llm_fast = scene_dict.get("llm_fast", {})
    llm_deep = scene_dict.get("llm_deep", {})
    # FAST and DEEP are independent — a scene can be fast-negative but deep-positive.
    if isinstance(llm_fast, dict) and llm_fast.get("objects_found"):
        lines.append("FAST:")
        for fnd in llm_fast.get("findings", []):
            lines.append(f"  [{fnd.get('type','?')}] {fnd.get('zone','?')} "
                         f"color={fnd.get('color','?')} conf={fnd.get('confidence',0)}")
            desc = fnd.get("description", "")
            if desc:
                lines.append(f"  {desc[:80]}")
    if isinstance(llm_deep, dict) and llm_deep.get("objects_found"):
        lines.append("DEEP:")
        for fnd in llm_deep.get("findings", []):
            lines.append(f"  [{fnd.get('type','?')}] {fnd.get('zone','?')} "
                         f"color={fnd.get('color','?')} conf={fnd.get('confidence',0)}")
            desc = fnd.get("description", "")
            if desc:
                lines.append(f"  {desc[:80]}")
    return lines


def build_llm_findings_preview(output_dir, report=None, scenes_dir=None):
    """Copy finding scenes to llm_findings/ and build preview_llm_findings.mp4.

    Args:
        output_dir: video output directory
        report: pre-loaded report dict (optional)
        scenes_dir: pre-resolved scenes dir (optional)
    """
    if scenes_dir is None:
        scenes_dir = os.path.join(output_dir, "scenes")

    if report is None:
        report_path = os.path.join(output_dir, "report.json")
        if not os.path.exists(report_path):
            print(f"  No report.json in {output_dir} — skipping")
            return
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

    scenes = report.get("scenes", [])
    fps = report.get("video_info", {}).get("fps", 30.0)

    print(f"\n{'='*60}")
    print("Copying scenes with LLM findings")
    print(f"{'='*60}")

    findings_scenes, preview_images = copy_llm_finding_scenes(scenes_dir, report)

    if not preview_images:
        print("  No images found to copy — skipping preview")
        return

    # Build scene_num -> scene_dict lookup for text overlay
    scene_by_num = {}
    for s in scenes:
        si = s.get("scene_id", 0)
        scene_by_num[si] = s

    # Build zone findings map
    zone_findings_map = build_scene_findings_map(scenes)

    # Text overlay function
    def findings_text_fn(scene_num, img_path):
        scene_dict = scene_by_num.get(scene_num, {})
        return _findings_text_for_scene(scene_dict, img_path)

    print(f"\n{'='*60}")
    print("Building llm_findings preview video")
    print(f"{'='*60}")

    images = [p for _, p in preview_images]
    build_preview_video(images, os.path.join(output_dir, "preview_llm_findings.mp4"),
                         fps=fps, duration_sec=PREVIEW_DURATION_FINDINGS,
                         zone_findings_map=zone_findings_map,
                         findings_text_fn=findings_text_fn,
                         label="llm_findings")


# ── Contact sheet ────────────────────────────────────────────────────

def build_contact_sheet(scenes, video_path, out_dir, fps=30.0,
                        enhance_fn=None, grid_fn=None):
    """Build contact_sheet.jpg — thumbnail grid of all scenes.

    Args:
        scenes: list of scene dicts (in-memory, with image_path_v3 etc.)
        video_path: source video path (for reading best frames)
        out_dir: output directory
        fps: video fps for timestamp display
        enhance_fn: optional fn(frame) -> enhanced frame (e.g. enhance_frame)
        grid_fn: optional fn(frame) -> gridded frame (e.g. draw_grid(frame, zones))
    """
    cap = cv2.VideoCapture(video_path)
    thumbs = []
    for si, scene in enumerate(scenes):
        fi = scene["best_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        if enhance_fn is not None:
            frame = enhance_fn(frame)
        if grid_fn is not None:
            frame = grid_fn(frame)
        thumb = cv2.resize(frame, (480, 270))
        llm = scene.get("llm_fast", {})
        found = isinstance(llm, dict) and llm.get("objects_found", False)
        conf = llm.get("confidence", 0) if isinstance(llm, dict) else 0
        color = (0, 0, 255) if found else (0, 255, 0) if conf > 0.3 else (128, 128, 128)
        cv2.putText(thumb, f"S{si:02d} f{fi} q={scene['best_score']:.0%}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(thumb, f"found={found} conf={conf:.0%}",
                    (5, 270 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        thumbs.append((si, thumb))
    cap.release()

    if not thumbs:
        return

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * 270, cols * 480, 3), dtype=np.uint8)
    for i, (si, thumb) in enumerate(thumbs):
        r, c = i // cols, i % cols
        sheet[r*270:(r+1)*270, c*480:(c+1)*480] = thumb
    cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  {len(thumbs)} scene thumbnails")
