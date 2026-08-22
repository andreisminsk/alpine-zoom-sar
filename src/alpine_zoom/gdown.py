"""Download files or folders from Google Drive, or videos from YouTube.

Usage:
  az-gdown file    <link> [-o OUTPUT]
  az-gdown dir     <link> [-o OUTPUT]
  az-gdown youtube <url>  [-o OUTPUT]

Downloads into the current folder unless -o/--output is specified.
"""
import sys
import os
import subprocess
import argparse

sys.stdout.reconfigure(encoding="utf-8")


def _download_file(link, output=None, quiet=False):
    """Download a single file from a Google Drive link.

    Share links (drive.google.com/file/d/ID/view) are parsed natively.
    Returns the path of the downloaded file, or None on failure.
    """
    import gdown
    if not gdown.parse_url.is_google_drive_url(link):
        print(f"  ERROR: not a valid Google Drive URL: {link}")
        return None
    out = gdown.download(link, output=output, quiet=quiet)
    return out


def _download_dir(link, output=None, quiet=False):
    """Download a Google Drive folder recursively.

    Returns the list of downloaded file paths, or None on failure.
    """
    import gdown
    if not gdown.parse_url.is_google_drive_url(link):
        print(f"  ERROR: not a valid Google Drive URL: {link}")
        return None
    out = gdown.download_folder(link, output=output, quiet=quiet)
    return out


def _download_youtube(url, output=None, quiet=False):
    """Download a YouTube video using yt-dlp.

    Uses the yt-dlp CLI via subprocess (more stable than the Python API).
    Returns the output path, or None on failure.
     """
    # Prefer the yt-dlp binary; fall back to the Python module.
    cmd = ["yt-dlp"]
    try:
        probe = subprocess.run(["yt-dlp", "--version"], capture_output=True)
    except FileNotFoundError:
        probe = None
    if probe is None or probe.returncode != 0:
        cmd = [sys.executable, "-m", "yt_dlp"]
        try:
            import yt_dlp   # noqa: F401
        except ImportError:
            print("  ERROR: yt-dlp not installed. Install with: pip install yt-dlp")
            return None

    out_dir = output or "."
    os.makedirs(out_dir, exist_ok=True)

    # -o with a template keeps the original filename; --no-playlist for single video
    cmd += ["-f", "bestvideo+bestaudio/best", "--no-playlist",
            "-o", os.path.join(out_dir, "%(title)s.%(ext)s")]
    if quiet:
        cmd.append("-q")
    cmd.append(url)

    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print("  ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return None

    if result.returncode != 0:
        err = (result.stderr or "").strip()[-500:]
        print(f"  ERROR: yt-dlp failed:\n{err}")
        return None

    if result.stdout and not quiet:
        for line in result.stdout.strip().split("\n")[-3:]:
            print(f"   {line}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description="Download a file or folder from Google Drive")
    sub = ap.add_subparsers(dest="kind", required=True)

    p_file = sub.add_parser("file", help="Download a single file")
    p_file.add_argument("link", help="Google Drive file link or ID")
    p_file.add_argument("-o", "--output", default=None,
                        help="Output path (default: current folder)")

    p_dir = sub.add_parser("dir", help="Download a folder recursively")
    p_dir.add_argument("link", help="Google Drive folder link or ID")
    p_dir.add_argument("-o", "--output", default=None,
                       help="Output directory (default: current folder)")

    p_yt = sub.add_parser("youtube", help="Download a YouTube video")
    p_yt.add_argument("url", help="YouTube video URL")
    p_yt.add_argument("-o", "--output", default=None,
                      help="Output directory (default: current folder)")

    args = ap.parse_args()

    if args.kind == "file":
        print(f"Downloading file: {args.link}")
        out = _download_file(args.link, output=args.output)
        if out:
            print(f"  Saved: {out}")
        else:
            print("  Download failed")
            sys.exit(1)
    elif args.kind == "dir":
        print(f"Downloading folder: {args.link}")
        out = _download_dir(args.link, output=args.output)
        if out:
            n = len(out) if isinstance(out, (list, tuple)) else 1
            dest = args.output or "."
            print(f"  Downloaded {n} file(s) to: {dest}")
        else:
            print("  Download failed")
            sys.exit(1)
    else:
        print(f"Downloading YouTube video: {args.url}")
        out = _download_youtube(args.url, output=args.output)
        if out:
            print(f"  Saved to: {out}")
        else:
            print("  Download failed")
            sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
