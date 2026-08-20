"""az-previews CLI — build preview videos from scene images."""
import sys
import os
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.previews import build_hq_lq_previews, build_color_anomalies_preview


def main():
    p = argparse.ArgumentParser(description="Build preview videos from scene images")
    p.add_argument("output_dir", help="Video output dir with report.json and scenes/")
    args = p.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: directory not found: {args.output_dir}")
        sys.exit(1)

    build_hq_lq_previews(args.output_dir)
    print()
    build_color_anomalies_preview(args.output_dir, os.path.join(args.output_dir, "scenes"))


if __name__ == "__main__":
    main()
