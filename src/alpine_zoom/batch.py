"""
Batch-run SAR scene analyzer on all videos in source_video/.

For each video, creates a dedicated output folder under analysis_results/
mirroring the source structure, e.g.:
  source_video/2026-08-11/DJI_xxx.MP4
  → analysis_results/2026-08-11/DJI_xxx.MP4/
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

# Default params (same as tested)
STRIDE = "dynamic"
QUALITY = 0.5
SCENE_SIM = 0.65
MAX_SCENES = 50
DEEP_TOP = 20
FAST_MODEL = "gemma4:31b-cloud"
DEEP_MODEL = "qwen3.5:397b-cloud"


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
    ap = argparse.ArgumentParser(description="Batch-run SAR scene analyzer on all videos")
    ap.add_argument("--context-file", default=None, dest="context_file",
                    help="Path to JSON mission context config. Passed to all videos.")
    ap.add_argument("--context-preset", default=None, dest="context_preset",
                    help="Named preset: 'sar', 'sar-heli'. Passed to all videos.")
    ap.add_argument("--llm-no-two-stage", action="store_true", dest="llm_no_two_stage",
                    help="Disable two-stage LLM mode (on by default). Use single-stage instead.")
    ap.add_argument("--llm-reasoning-model", default="glm-5.1:cloud", dest="llm_reasoning_model",
                    help="Reasoning model for two-stage mode (default: glm-5.1:cloud).")
    ap.add_argument("--llm-pipeline", default="fast", choices=["fast", "chancepeek", "max"],
                    help="LLM pipeline mode: 'fast' (default), 'chancepeek' (deep on fast negatives), 'max' (all variants).")
    ap.add_argument("--llm-parallel", type=int, default=0, dest="llm_parallel",
                    help="Number of parallel LLM workers (default 0 = sequential). 4 recommended for cloud models.")
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
            "--stride", str(STRIDE),
            "--quality", str(QUALITY),
            "--scene-sim", str(SCENE_SIM),
            "--llm-scenes-cap", str(MAX_SCENES),
            "--llm-deep-max-scenes", str(DEEP_TOP),
            "--llm-fast-model", FAST_MODEL,
            "--llm-deep-model", DEEP_MODEL,
            "--llm-run",
            "--llm-pipeline", "fast",
            "--llm-parallel", "0",
            "--color-anomalies",
            "--build-preview",
        ]
        if is_helicopter:
            cmd.append("--helicopter")
        if args.context_file:
            cmd.extend(["--context-file", args.context_file])
        elif args.context_preset:
            cmd.extend(["--context-preset", args.context_preset])
        if not args.llm_no_two_stage:
            # Two-stage is on by default — pass --llm-reasoning-model
            cmd.extend(["--llm-reasoning-model", args.llm_reasoning_model])
        else:
            cmd.append("--llm-no-two-stage")
        if args.llm_pipeline != "fast":
            cmd.extend(["--llm-pipeline", args.llm_pipeline])
        if args.llm_parallel > 0:
            cmd.extend(["--llm-parallel", str(args.llm_parallel)])

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
