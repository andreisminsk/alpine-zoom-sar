"""Report information about a video file.

Usage:
  az-info <video_path> [video_path2 ...]

Example:
  az-info source_video/VIDEO.HELI/C0003.MP4
  az-info source_video/VIDEO.DRONE/*.MP4
"""
import sys
import os
import glob
import argparse
import cv2

sys.stdout.reconfigure(encoding="utf-8")


def get_video_info(path):
    """Extract video info using OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    info = {
        "path": path,
        "filename": os.path.basename(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": 0,
        "duration_str": "",
        "size_mb": 0,
    }
    cap.release()

    if info["fps"] > 0:
        info["duration_sec"] = round(info["frame_count"] / info["fps"], 1)
        h = int(info["duration_sec"] // 3600)
        m = int((info["duration_sec"] % 3600) // 60)
        s = info["duration_sec"] % 60
        info["duration_str"] = f"{h:02d}:{m:02d}:{s:05.2f}"

    if os.path.exists(path):
        info["size_mb"] = round(os.path.getsize(path) / 1e6, 1)

    # Check for companion files
    base, _ = os.path.splitext(path)
    companions = []
    for ext in [".SRT", ".srt", "M01.XML", "M01.xml", ".XML", ".xml"]:
        cpath = base + ext
        if os.path.exists(cpath):
            companions.append(os.path.basename(cpath))
    info["companions"] = companions

    return info


def main():
    p = argparse.ArgumentParser(description="Report information about video files")
    p.add_argument("videos", nargs="+", help="Video path(s), supports wildcards")
    args = p.parse_args()

     # Expand wildcards
    paths = []
    for arg in args.videos:
        expanded = glob.glob(arg)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(arg):
            paths.append(arg)
        else:
            print(f"  WARNING: not found: {arg}")

    for path in paths:
        info = get_video_info(path)
        if info is None:
            print(f"  ERROR: cannot open {path}")
            continue

        print(f"{'='*60}")
        print(f"File:      {info['filename']}")
        print(f"Path:      {info['path']}")
        print(f"Size:      {info['size_mb']:.1f} MB")
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"FPS:       {info['fps']}")
        print(f"Frames:    {info['frame_count']}")
        print(f"Duration:  {info['duration_str']} ({info['duration_sec']}s)")
        if info["companions"]:
            print(f"Companions: {', '.join(info['companions'])}")
        else:
            print(f"Companions: none")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
