"""Generate enhancement variants from a single source image, with optional LLM analysis.

Produces _grid_v1.jpg, _grid_v2.jpg, _grid_v3.jpg, _grid_v4.jpg
in the same folder as the source image.
Optionally runs LLM analysis on each variant (like the main pipeline).

Usage:
  python image_analyzer.py <image_path> [image_path2 ...]
  python image_analyzer.py <image_path> --no-llm
  python image_analyzer.py <image_path> --fast-model gemma4:31b-cloud --deep-model qwen3.5:397b-cloud

Example:
  python image_analyzer.py scene_32_f00810_orig.jpg
  python image_analyzer.py scene_32_f00810_orig.jpg --no-llm
"""
import sys
import os
import json
import time
import argparse
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.common import zone_grid, draw_grid
from alpine_zoom.llm import (
    build_prompt, llm_analyze, llm_text_analyze, resolve_mission_context,
)

# Enhancement functions stay imported from alpine_zoom.video (source of truth)
# since they are pipeline-specific and not yet extracted to a shared module.
from alpine_zoom.video import (
    enhance_frame_v1, enhance_frame_v2, enhance_frame_v3, enhance_frame_v4,
)


def main():
    p = argparse.ArgumentParser(description="Generate enhancement variants + optional LLM analysis")
    p.add_argument("images", nargs="+", help="Source image path(s)")
    p.add_argument("--llm-run", action="store_true", help="Run LLM analysis (off by default — variants only unless specified)")
    p.add_argument("--llm-fast-model", default="gemma4:31b-cloud", help="Fast LLM model (default: gemma4:31b-cloud)")
    p.add_argument("--llm-deep-model", default="qwen3.5:397b-cloud", help="Deep LLM model (default: qwen3.5:397b-cloud)")
    p.add_argument("--llm-no-deep", action="store_true", help="Skip deep LLM analysis (auto-triggered by default)")
    p.add_argument("--llm-force-deep", action="store_true",
                   help="Run deep LLM on ALL variants (not just fast-LLM positives)")
    p.add_argument("--helicopter", action="store_true", help="Use helicopter-mode prompt")
    p.add_argument("--context-file", default=None, dest="context_file",
                   help="Path to JSON mission context config.")
    p.add_argument("--context-preset", default=None, dest="context_preset",
                   help="Named preset: 'sar', 'sar-heli'.")
    p.add_argument("--llm-timeout", type=int, default=120, help="LLM timeout in seconds")
    args = p.parse_args()

    variants = [
        ("orig", None),  # original, no enhancement
        ("v1", enhance_frame_v1),
        ("v2", enhance_frame_v2),
        ("v3", enhance_frame_v3),
        ("v4", enhance_frame_v4),
    ]

    # Resolve mission context (fixes: previously ignored context even with --helicopter)
    mission_context, ctx_source = resolve_mission_context(
        context_file=args.context_file,
        context_preset=args.context_preset,
        helicopter=args.helicopter,
    )
    prompt = build_prompt(helicopter=args.helicopter, mission_context=mission_context)
    if args.llm_run:
        print(f"Mission context: {ctx_source}")

    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ERROR: cannot read {img_path}")
            continue

        base, ext = os.path.splitext(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(os.path.dirname(img_path) or ".", img_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing: {os.path.basename(img_path)} -> {out_dir}")

        h, w = img.shape[:2]
        zones = zone_grid(h, w)

        variant_paths = []
        for name, func in variants:
            result = img.copy() if func is None else func(img)
            gridded = draw_grid(result, zones)
            out_path = os.path.join(out_dir, f"{img_name}_grid_{name}.jpg")
            cv2.imwrite(out_path, gridded, [cv2.IMWRITE_JPEG_QUALITY, 95])
            variant_paths.append((name, out_path))
            print(f"  {name}: {out_path}")

        if args.llm_run:
            llm_log = []
            llm_log.append(f"LLM Analysis Report")
            llm_log.append(f"Source: {os.path.basename(img_path)}")
            llm_log.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            llm_log.append(f"Fast model: {args.llm_fast_model}")
            if not args.llm_no_deep:
                llm_log.append(f"Deep model: {args.llm_deep_model}")
            llm_log.append(f"Mission context: {ctx_source}")
            llm_log.append("")

            print(f"\n  LLM Analysis (fast: {args.llm_fast_model})")
            all_findings = []
            llm_available = True
            for name, path in variant_paths:
                t0 = time.time()
                result, elapsed, error = llm_analyze(
                    args.llm_fast_model, path, prompt, timeout=args.llm_timeout)
                if error:
                    print(f"    {name}: ERROR ({elapsed:.1f}s) {error}")
                    llm_log.append(f"[FAST] {name}: ERROR ({elapsed:.1f}s) {error}")
                    llm_available = False
                elif result:
                    found = result.get("objects_found", False)
                    conf = result.get("confidence", 0)
                    n_findings = len(result.get("findings", []))
                    print(f"    {name}: found={found} conf={conf} findings={n_findings} ({elapsed:.1f}s)")
                    llm_log.append(f"[FAST] {name}: found={found} conf={conf} findings={n_findings} ({elapsed:.1f}s)")
                    if found:
                        for fnd in result.get("findings", []):
                            line = (f"  → {fnd.get('type','?')} {fnd.get('zone','')} "
                                    f"color={fnd.get('color','?')} conf={fnd.get('confidence',0)} "
                                    f"{fnd.get('description','')}")
                            print(f"      {line[:100]}")
                            llm_log.append(f"    {line}")
                        all_findings.append((name, result))
                llm_log.append("")

            # Deep LLM on positives (auto-triggered when any fast variant finds something)
            # or on ALL variants if --llm-force-deep
            deep_variants = all_findings if not args.llm_force_deep else variant_paths
            if not args.llm_no_deep and deep_variants and llm_available:
                print(f"\n  Deep LLM Analysis ({args.llm_deep_model})"
                      + (" [ALL variants]" if args.llm_force_deep else ""))
                llm_log.append(f"Deep LLM Analysis ({args.llm_deep_model})"
                               + (" [ALL variants]" if args.llm_force_deep else ""))
                llm_log.append("")
                deep_results = []
                for item in deep_variants:
                    name = item[0]
                    path = os.path.join(out_dir, f"{img_name}_grid_{name}.jpg")
                    result, elapsed, error = llm_analyze(
                        args.llm_deep_model, path, prompt, timeout=args.llm_timeout * 3)
                    if error:
                        print(f"    {name}: ERROR ({elapsed:.1f}s) {error}")
                        llm_log.append(f"[DEEP] {name}: ERROR ({elapsed:.1f}s) {error}")
                    elif result:
                        found = result.get("objects_found", False)
                        conf = result.get("confidence", 0)
                        print(f"    {name}: found={found} conf={conf} ({elapsed:.1f}s)")
                        llm_log.append(f"[DEEP] {name}: found={found} conf={conf} ({elapsed:.1f}s)")
                        for fnd in result.get("findings", []):
                            line = (f"  → {fnd.get('type','?')} {fnd.get('zone','')} "
                                    f"color={fnd.get('color','?')} conf={fnd.get('confidence',0)} "
                                    f"{fnd.get('description','')}")
                            print(f"      {line[:100]}")
                            llm_log.append(f"    {line}")
                        deep_results.append((name, result))
                    llm_log.append("")

                # Overall summary using deep LLM
                if deep_results:
                    print(f"\n  Overall Summary ({args.llm_deep_model})")
                    llm_log.append(f"Overall Summary ({args.llm_deep_model})")
                    llm_log.append("")

                    # Build summary prompt with all findings
                    summary_input = "You are a SAR analyst. Below are findings from multiple enhancement variants of the same image.\n"
                    summary_input += "The [DEEP] results are from a more capable model and should be weighted more heavily than [FAST] results.\n"
                    summary_input += "Provide an overall assessment: is there a climber/gear? Which zone? Confidence? What is the most likely explanation?\n\n"
                    for name, result in deep_results:
                        summary_input += f"[DEEP] Variant {name}:\n"
                        for fnd in result.get("findings", []):
                            summary_input += f"  {fnd.get('type','?')} {fnd.get('zone','')} color={fnd.get('color','?')} conf={fnd.get('confidence',0)}: {fnd.get('description','')}\n"
                    summary_input += "\nProvide a concise overall verdict in 3-5 sentences. "
                    summary_input += "Start with 'Found it.' if you believe a climber or gear is present, or 'No Detection.' if not."

                    summary_result, s_elapsed, s_error = llm_text_analyze(
                        args.llm_deep_model, summary_input, timeout=args.llm_timeout * 3)
                    if s_error:
                        print(f"    Summary: ERROR ({s_elapsed:.1f}s) {s_error}")
                        llm_log.append(f"[SUMMARY] ERROR ({s_elapsed:.1f}s) {s_error}")
                    else:
                        print(f"    Summary ({s_elapsed:.1f}s):")
                        for line in summary_result.strip().split("\n"):
                            print(f"      {line}")
                            llm_log.append(f"  {line}")
                    llm_log.append("")

            # Note if LLM was unavailable
            if not llm_available:
                llm_log.append("NOTE: LLM was unavailable (Ollama not running or model not found).")
                llm_log.append("Variants were generated successfully. Re-run without --no-llm to retry LLM analysis.")
                print(f"\n  NOTE: LLM unavailable — variants only. See llm_analysis.txt for details.")

            # Write LLM analysis log
            log_path = os.path.join(out_dir, "llm_analysis.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(llm_log))
            print(f"\n  LLM analysis saved: {log_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
