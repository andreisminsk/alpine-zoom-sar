"""Summarize findings from one or more report.json files.

Usage:
  az-report <report_path> [report_path2 ...]
  az-report analysis_results/.../C0003.MP4/report.json
  az-report analysis_results/VIDEO.HELI/*/report.json
"""
import sys
import os
import json
import glob
import argparse

sys.stdout.reconfigure(encoding="utf-8")


def summarize_report(path):
    """Print summary of a single report.json."""
    with open(path, encoding="utf-8") as f:
        r = json.load(f)

    video = os.path.basename(r.get("video", path))
    vi = r.get("video_info", {})
    print(f"{'='*80}")
    print(f"Video: {video}")
    print(f"  Resolution: {vi.get('width','?')}x{vi.get('height','?')} @ {vi.get('fps','?')}fps")
    print(f"  Duration: {vi.get('duration_sec','?')}s, {vi.get('total_frames','?')} frames")
    print(f"  Scenes: {r.get('total_scenes','?')}, Quality frames: {r.get('quality_frames','?')}")
    rt = r.get("recording_time")
    if rt:
        print(f"  Recording time: {rt}")
    if "analysis_elapsed_sec" in r:
        print(f"  Analysis: {r.get('analysis_start','?')} → {r.get('analysis_end','?')} ({r.get('analysis_elapsed_sec',0):.0f}s)")

    fast_pos = []
    deep_pos = []
    for s in r.get("scenes", []):
        sid = s.get("scene_id", 0)
        bf = s.get("best_frame", 0)
        ts = s.get("time_str", "")
        lf = s.get("llm_fast", {})
        ld = s.get("llm_deep", {})
        if isinstance(lf, dict) and lf.get("objects_found"):
            fast_pos.append((sid, bf, ts, lf))
        if isinstance(ld, dict) and ld.get("objects_found"):
            deep_pos.append((sid, bf, ts, ld))

    print(f"\n  Fast model positives: {len(fast_pos)}")
    print(f"  Deep model positives: {len(deep_pos)}")

    if fast_pos:
        print(f"\n  {'─'*76}")
        print(f"  FAST MODEL FINDINGS")
        print(f"  {'─'*76}")
        for sid, bf, ts, lf in fast_pos:
            print(f"\n  Scene {sid:04d} | f{bf:05d} | t={ts} | conf={lf.get('confidence',0)}")
            for fnd in lf.get("findings", []):
                print(f"    [{fnd.get('type','?')}] {fnd.get('zone','?')} "
                      f"color={fnd.get('color','?')} conf={fnd.get('confidence',0)}")
                print(f"      {fnd.get('description','')[:120]}")

    if deep_pos:
        print(f"\n  {'─'*76}")
        print(f"  DEEP MODEL FINDINGS (confirmed)")
        print(f"  {'─'*76}")
        for sid, bf, ts, ld in deep_pos:
            print(f"\n  Scene {sid:04d} | f{bf:05d} | t={ts} | conf={ld.get('confidence',0)}")
            for fnd in ld.get("findings", []):
                print(f"    [{fnd.get('type','?')}] Z={fnd.get('zone','?')} "
                      f"color={fnd.get('color','?')} conf={fnd.get('confidence',0)}")
                print(f"      {fnd.get('description','')[:150]}")
            summary = ld.get("summary")
            if summary:
                print(f"  SUMMARY: {summary[:200]}")

    # Non-orange findings (potential other gear/climbers)
    non_orange = []
    for sid, bf, ts, lf in fast_pos:
        for fnd in lf.get("findings", []):
            c = fnd.get("color", "?")
            if "orange" not in c.lower():
                non_orange.append((sid, bf, ts, fnd))
    if non_orange:
        print(f"\n  {'─'*76}")
        print(f"  NON-ORANGE FINDINGS (possible other gear/climbers)")
        print(f"  {'─'*76}")
        for sid, bf, ts, fnd in non_orange:
            print(f"    Scene {sid:04d} f{bf:05d} t={ts} "
                  f"[{fnd.get('type','?')}] {fnd.get('zone','?')} "
                  f"color={fnd.get('color','?')} conf={fnd.get('confidence',0)}")
            print(f"      {fnd.get('description','')[:120]}")

    print()


def main():
    p = argparse.ArgumentParser(description="Summarize findings from report.json files")
    p.add_argument("reports", nargs="+", help="Report path(s), supports wildcards")
    args = p.parse_args()

    # Expand wildcards
    paths = []
    for arg in args.reports:
        expanded = glob.glob(arg)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(arg):
            paths.append(arg)
        else:
            print(f"  WARNING: not found: {arg}")

    if not paths:
        print("No report files found.")
        sys.exit(1)

    paths.sort()
    print(f"Summarizing {len(paths)} report(s)\n")

    total_fast = 0
    total_deep = 0
    for path in paths:
        try:
            summarize_report(path)
        except Exception as e:
            print(f"  ERROR reading {path}: {e}")

    print(f"{'='*80}")
    print(f"Done! {len(paths)} report(s) summarized.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
