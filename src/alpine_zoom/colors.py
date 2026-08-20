"""Color anomaly detection — find small colored regions unusual for the scene.

Scene-relative approach: builds a color distribution from the image itself,
then flags small clusters of pixels whose color is statistically rare.
No predefined color targets — the terrain defines "normal".

This is now a thin CLI wrapper around alpine_zoom.color (canonical implementation).

Usage:
  python color_anomalies.py <image_path> [--debug]
  python color_anomalies.py <scenes_dir> [--debug]

Example:
  python color_anomalies.py analysis_results/.../scene_352_f01890_orig.jpg --debug
"""
import sys
import os
import json
import cv2
import numpy as np
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.color import detect_color_anomalies, draw_color_findings
from alpine_zoom.common import (
    draw_text_with_bg, build_dynamics, canonical_base, collect_orig_images,
)


def process_image(path, debug=False, out_dir=None):
    """Process a single image.

    Produces canonical output matching az-video:
        - filename: <scene_base>_color_anomalies.jpg (no '_orig' suffix)
        - findings-count text marker (same style/position as az-video)

    Returns the findings list (empty if none / unreadable).
    """
    img = cv2.imread(path)
    if img is None:
        print(f"  ERROR: cannot read {path}")
        return []

    h, w = img.shape[:2]
    base = canonical_base(path)

    if debug:
        findings, anomaly_mask, colorfulness, rarity = detect_color_anomalies(img, debug=True)
    else:
        findings = detect_color_anomalies(img, debug=False)

    # Match az-video: only print per-scene when findings exist (no "Findings: 0" noise).
    if findings:
        print(f"\n   {os.path.basename(path)} ({w}x{h})")
        print(f"  Findings: {len(findings)}")
        for f in findings:
            print(f"     {f['color']} Z{f['zone']} area={f['area']} conf={f['confidence']} "
                  f"colorful={f['colorfulness']} rarity={f['rarity']} "
                  f"bbox={f['bbox']} lab=({f['avg_lab'][0]},{f['avg_lab'][1]})")

    # Save result
    if out_dir is None:
        out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if findings:
        result = draw_color_findings(img, findings)
        # Add findings count below existing text markers (same style as az-video)
        n_findings = len(findings)
        label = f"{n_findings} finding{'s' if n_findings != 1 else ''}"
        draw_text_with_bg(result, label, 10, 116, cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, 1, (0, 255, 255))
        out_path = os.path.join(out_dir, f"{base}_color_anomalies.jpg")
        cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved: {out_path}")

    if debug:
        # Save anomaly mask heatmap
        hm = cv2.applyColorMap(anomaly_mask * 255, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(img, 0.6, hm, 0.4, 0)
        out_path = os.path.join(out_dir, f"{base}_anomaly_mask.jpg")
        cv2.imwrite(out_path, blend, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Debug: anomaly mask -> {out_path}")

        # Save colorfulness heatmap
        cf_norm = np.clip(colorfulness / 100 * 255, 0).astype(np.uint8)
        hm2 = cv2.applyColorMap(cf_norm, cv2.COLORMAP_JET)
        blend2 = cv2.addWeighted(img, 0.6, hm2, 0.4, 0)
        out_path = os.path.join(out_dir, f"{base}_colorfulness.jpg")
        cv2.imwrite(out_path, blend2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Debug: colorfulness -> {out_path}")

        # Save rarity heatmap
        if rarity is not None:
            rar_norm = np.clip(rarity / 10 * 255, 0).astype(np.uint8)
            hm3 = cv2.applyColorMap(rar_norm, cv2.COLORMAP_JET)
            blend3 = cv2.addWeighted(img, 0.6, hm3, 0.4, 0)
            out_path = os.path.join(out_dir, f"{base}_rarity.jpg")
            cv2.imwrite(out_path, blend3, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Debug: rarity -> {out_path}")

    return findings


def update_report(report_path, results):
    """Write color findings back into report.json (dynamics.color_findings).

    `results` maps canonical base (e.g. 'scene_05_f01101') -> findings list.
    Uses the shared build_dynamics() so the structure matches az-video exactly.
    """
    if not os.path.exists(report_path):
        print(f"  No report.json at {report_path} — skipping update")
        return
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    updated = 0
    for s in report.get("scenes", []):
        si = s.get("scene_id", 0)
        fi = s.get("best_frame", 0)
        base = f"scene_{si:02d}_f{fi:05d}"
        findings = results.get(base)
        if findings is None:
            continue
        s["dynamics"] = build_dynamics(
            findings,
            image_file=f"{base}_color_anomalies.jpg" if findings else "",
        )
        updated += 1
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
     # az-video writes dynamics for every analyzed scene (empty or not); mirror that.
    with_findings = sum(1 for v in results.values() if v)
    print(f"  Color anomalies: {updated} scenes ({with_findings} with findings) "
          f"-> {report_path}")


def main():
    p = argparse.ArgumentParser(description="Color anomaly detection (scene-relative)")
    p.add_argument("output_dir",
                   help="Video output dir with report.json and scenes/ (like az-previews)")
    p.add_argument("--debug", action="store_true", help="Save debug heatmaps")
    p.add_argument("--skip-update-report", action="store_true",
                   help="Do not write dynamics.color_findings back into report.json")
    args = p.parse_args()

    scenes_dir = os.path.join(args.output_dir, "scenes")
    anomalies_dir = os.path.join(scenes_dir, "anomalies")
    os.makedirs(anomalies_dir, exist_ok=True)

    images = collect_orig_images(scenes_dir)
    print(f"Found {len(images)} _orig.jpg images in {scenes_dir}")

    results = {}      # canonical base -> findings
    for img_path in images:
        findings = process_image(img_path, debug=args.debug, out_dir=anomalies_dir)
        results[canonical_base(img_path)] = findings

    if not args.skip_update_report:
        report_path = os.path.join(args.output_dir, "report.json")
        update_report(report_path, results)

    print("\nDone!")


if __name__ == "__main__":
    main()
