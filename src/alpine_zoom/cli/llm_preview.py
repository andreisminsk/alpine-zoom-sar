"""az-llm-preview CLI — build LLM findings preview videos."""
import sys
import os
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.previews import build_llm_findings_preview
from alpine_zoom.common import find_video_dirs


def main():
    p = argparse.ArgumentParser(description="Build LLM findings preview videos")
    p.add_argument("results_root", help="Root dir containing video analysis subdirs")
    args = p.parse_args()

    if not os.path.isdir(args.results_root):
        print(f"ERROR: directory not found: {args.results_root}")
        sys.exit(1)

    video_dirs = find_video_dirs(args.results_root)
    print(f"Found {len(video_dirs)} video analysis dirs")

    for video_dir in video_dirs:
        build_llm_findings_preview(video_dir)

    print(f"\nDone! Processed {len(video_dirs)} video(s)")


if __name__ == "__main__":
    main()
