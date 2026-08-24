"""
Batch-run SAR scene analyzer on all videos in source_video/.

For each video, creates a dedicated output folder under analysis_results/
mirroring the source structure, e.g.:
  source_video/2026-08-11/DJI_xxx.MP4
  → analysis_results/2026-08-11/DJI_xxx.MP4/

All az-video parameters are available as az-batch arguments and passed through.
Defaults match az-batch's tuned values (e.g. --scene-sim 0.65) rather than
az-video's single-video defaults.
"""
import sys
import os
import subprocess
import time
import argparse

sys.stdout.reconfigure(encoding="utf-8")

SOURCE_ROOT = "source_video"
RESULTS_ROOT = "analysis_results"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def find_videos(root):
    """Walk root dir, return list of (full_path, rel_path)."""
    videos = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                videos.append((full, rel))
    videos.sort()
    return videos


def main():
    ap = argparse.ArgumentParser(
        description="Batch-run SAR scene analyzer on all videos",
        epilog="All parameters are passed through to az-video. "
               "Defaults are tuned for batch processing (e.g. --scene-sim 0.65).")
    # ── Video processing ──────────────────────────────────────────────
    ap.add_argument("--stride", default="dynamic",
                    help="Frame sampling interval: integer or 'dynamic' (default: dynamic)")
    ap.add_argument("--quality", type=float, default=0.5,
                    help="Quality threshold 0-1 (default: 0.5)")
    ap.add_argument("--scene-sim", type=float, default=0.65,
                    help="Scene similarity threshold 0-1 (default: 0.65, more aggressive than az-video's 0.82)")
    ap.add_argument("--dedup-thresh", type=float, default=0.90,
                    help="Scene deduplication threshold (default: 0.90, set 0 to disable)")
    # ── LLM ───────────────────────────────────────────────────────────
    ap.add_argument("--llm-scenes-cap", type=int, default=50, dest="llm_scenes_cap",
                    help="Max scenes to send to fast LLM (default: 50)")
    ap.add_argument("--llm-deep-max-scenes", type=int, default=20, dest="llm_deep_max_scenes",
                    help="Max scenes to send to deep LLM (default: 20)")
    ap.add_argument("--llm-fast-model", default="gemma4:31b-cloud", dest="llm_fast_model",
                    help="Fast LLM model (default: gemma4:31b-cloud)")
    ap.add_argument("--llm-deep-model", default="qwen3.5:397b-cloud", dest="llm_deep_model",
                    help="Deep LLM model (default: qwen3.5:397b-cloud)")
    ap.add_argument("--llm-pipeline", default="fast", choices=["fast", "chancepeek", "max"],
                    help="LLM pipeline mode (default: fast)")
    ap.add_argument("--llm-parallel", type=int, default=0, dest="llm_parallel",
                    help="Number of parallel LLM workers (default: 0 = sequential, 4 recommended for cloud)")
    ap.add_argument("--llm-no-two-stage", action="store_true", dest="llm_no_two_stage",
                    help="Disable two-stage LLM mode (on by default)")
    ap.add_argument("--llm-reasoning-model", default="glm-5.1:cloud", dest="llm_reasoning_model",
                    help="Reasoning model for two-stage mode (default: glm-5.1:cloud)")
    # ── Context ───────────────────────────────────────────────────────
    ap.add_argument("--context-file", default=None, dest="context_file",
                    help="Path to JSON mission context config. Passed to all videos.")
    ap.add_argument("--context-preset", default=None, dest="context_preset",
                    help="Named preset: 'sar', 'sar-heli'. Passed to all videos.")
    # ── Output options ─────────────────────────────────────────────────
    ap.add_argument("--no-llm-run", action="store_true", dest="no_llm_run",
                    help="Do NOT run LLM analysis (generate scenes/images only)")
    ap.add_argument("--no-build-preview", action="store_true", dest="no_build_preview",
                    help="Do NOT build preview videos")
    ap.add_argument("--enforce-thresholds", action="store_true", dest="enforce_thresholds",
                    help="Do not auto-relax quality/scene thresholds for low-quality video. "
                         "By default, if very few frames pass quality scan, thresholds are "
                         "relaxed (quality→0.3, scene-sim→0.55) and the scan is re-run.")
    ap.add_argument("--no-color-anomalies", action="store_true", dest="no_color_anomalies",
                    help="Do NOT run color anomaly detection")
    args = ap.parse_args()

    if not os.path.isdir(SOURCE_ROOT):
        print(f"ERROR: source folder not found: {SOURCE_ROOT}")
        sys.exit(1)

    videos = find_videos(SOURCE_ROOT)
    if not videos:
        print(f"No videos found in {SOURCE_ROOT}")
        sys.exit(0)

    print(f"Found {len(videos)} video(s) to analyze")
    print(f"Output root: {RESULTS_ROOT}")
    if args.context_file:
        print(f"Context file: {args.context_file}")
    elif args.context_preset:
        print(f"Context preset: {args.context_preset}")
    print()

    for i, (full_path, rel_path) in enumerate(videos):
        # Output dir mirrors source structure
        out_dir = os.path.join(RESULTS_ROOT, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        # Skip if already analyzed (report.json exists)
        report_path = os.path.join(out_dir, "report.json")
        if os.path.exists(report_path):
            print(f"[{i+1}/{len(videos)}] SKIP (already analyzed): {rel_path}")
            continue

        print(f"[{i+1}/{len(videos)}] Analyzing: {rel_path}")
        print(f"  Output: {out_dir}")
        t0 = time.time()

        # Auto-detect helicopter footage by folder name
        is_helicopter = "HELI" in rel_path.upper() or "helicopter" in rel_path.lower()
        if is_helicopter:
            print(f"  HELICOPTER MODE detected (folder name contains HELI)")

        cmd = [
            sys.executable, "-m", "alpine_zoom.cli.video",
            full_path,
            "-o", out_dir,
            "--stride", str(args.stride),
            "--quality", str(args.quality),
            "--scene-sim", str(args.scene_sim),
            "--dedup-thresh", str(args.dedup_thresh),
            "--llm-scenes-cap", str(args.llm_scenes_cap),
            "--llm-deep-max-scenes", str(args.llm_deep_max_scenes),
            "--llm-fast-model", args.llm_fast_model,
            "--llm-deep-model", args.llm_deep_model,
            "--llm-pipeline", args.llm_pipeline,
        ]
        # LLM run (on by default)
        if not args.no_llm_run:
            cmd.append("--llm-run")
        # Preview (on by default)
        if not args.no_build_preview:
            cmd.append("--build-preview")
        # Color anomalies (on by default)
        if not args.no_color_anomalies:
            cmd.append("--color-anomalies")
        # Helicopter mode
        if is_helicopter:
            cmd.append("--helicopter")
        # Context
        if args.context_file:
            cmd.extend(["--context-file", args.context_file])
        elif args.context_preset:
            cmd.extend(["--context-preset", args.context_preset])
        # Two-stage LLM
        if not args.llm_no_two_stage:
            cmd.extend(["--llm-reasoning-model", args.llm_reasoning_model])
        else:
            cmd.append("--llm-no-two-stage")
        # Parallel
        if args.llm_parallel > 0:
            cmd.extend(["--llm-parallel", str(args.llm_parallel)])
        # Enforce thresholds
        if args.enforce_thresholds:
            cmd.append("--enforce-thresholds")

        result = None
        try:
            result = subprocess.run(cmd, timeout=3600,  # 60 min per video
                                    capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f"  DONE ({elapsed:.0f}s)")
            else:
                print(f"  FAILED ({elapsed:.0f}s, exit {result.returncode})")
                print(f"  stderr: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (60 min) — partial results may exist")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Print last few lines of stdout for progress
        if result and result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")

        print()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    positives_total = 0
    for full_path, rel_path in videos:
        out_dir = os.path.join(RESULTS_ROOT, rel_path)
        report_path = os.path.join(out_dir, "report.json")
        if not os.path.exists(report_path):
            print(f"  {rel_path}: NOT ANALYZED")
            continue
        import json
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        n_scenes = report.get("total_scenes", 0)
        scenes = report.get("scenes", [])
        pos = [s for s in scenes
               if s.get("llm_fast", {}).get("objects_found", False)]
        deep_pos = [s for s in scenes
                     if s.get("llm_deep", {}).get("objects_found", False)]
        status = "POSITIVES" if pos else "clear"
        print(f"  {rel_path}: {n_scenes} scenes, "
              f"{len(pos)} fast-pos, {len(deep_pos)} deep-pos [{status}]")
        if pos:
            positives_total += len(pos)
            for s in pos:
                llm = s.get("llm_fast", {})
                print(f"    → f{s['best_frame']} t={s.get('time_str','')} "
                      f"conf={llm.get('confidence',0)}")
                for fnd in llm.get("findings", []):
                    print(f"      {fnd.get('type','?')} {fnd.get('zone','')} "
                          f"color={fnd.get('color','?')} "
                          f"{fnd.get('description','')[:80]}")

    print(f"\nTotal positives across all videos: {positives_total}")
    print(f"Results: {RESULTS_ROOT}/")


if __name__ == "__main__":
    main()
