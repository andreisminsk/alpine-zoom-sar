"""Re-run LLM analysis on existing scene images without regenerating them.
Updates report.json with new LLM results.

Usage:
    python llm_analysis.py <video_output_dir> [options]

Example:
    python llm_analysis.py analysis_results/2026-08-15/VIDEO.MP4 --helicopter
    python llm_analysis.py analysis_results/2026-08-15/VIDEO.MP4 --llm-force-deep
"""
import sys
import os
import json
import time
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.llm import (
    build_prompt, llm_analyze, two_stage_analyze, resolve_mission_context,
    parallel_llm_batch,
)
from alpine_zoom.common import find_scene_image, find_all_variants, load_report


def main():
    p = argparse.ArgumentParser(description="Re-run LLM on existing scene images")
    p.add_argument("output_dir", help="Video output dir with report.json and scenes/")
    p.add_argument("--helicopter", action="store_true")
    p.add_argument("--llm-fast-model", default="gemma4:31b-cloud")
    p.add_argument("--llm-deep-model", default="qwen3.5:397b-cloud")
    p.add_argument("--llm-deep-max-scenes", type=int, default=20)
    p.add_argument("--llm-fast-max-scenes", type=int, default=100)
    p.add_argument("--llm-force-deep", action="store_true",
                   help="Run deep LLM on ALL scenes (not just fast-LLM positives)")
    p.add_argument("--llm-pipeline", default="fast", choices=["fast", "chancepeek", "max"],
                   help="LLM pipeline mode: 'fast' (default), 'chancepeek' (deep on fast negatives), 'max' (all variants).")
    p.add_argument("--context-file", default=None, dest="context_file",
                   help="Path to JSON mission context config. Overrides preset/helicopter/stored context.")
    p.add_argument("--context-preset", default=None, dest="context_preset",
                   help="Named preset: 'sar', 'sar-heli'. Overrides --helicopter and stored context.")
    p.add_argument("--llm-no-two-stage", action="store_true", dest="llm_no_two_stage",
                   help="Disable two-stage LLM mode (on by default). Use single-stage instead.")
    p.add_argument("--llm-reasoning-model", default="glm-5.1:cloud", dest="llm_reasoning_model",
                   help="Reasoning model for two-stage mode (default: glm-5.1:cloud).")
    p.add_argument("--llm-parallel", type=int, default=0, dest="llm_parallel",
                   help="Number of parallel LLM workers (default 0 = sequential). 4 recommended for cloud models.")
    args = p.parse_args()

    report, scenes_dir = load_report(args.output_dir)
    report_path = os.path.join(args.output_dir, "report.json")
    scenes = report["scenes"]

    # Resolve mission context: --context-file > --context-preset > stored in report > --helicopter > none
    mission_context, ctx_source = resolve_mission_context(
        context_file=args.context_file,
        context_preset=args.context_preset,
        helicopter=args.helicopter,
        report=report,
    )
    prompt = build_prompt(helicopter=args.helicopter, mission_context=mission_context)

    print(f"Re-running LLM analysis on {len(scenes)} scenes")
    print(f"Fast model: {args.llm_fast_model}")
    print(f"Deep model: {args.llm_deep_model}")
    print(f"Mission context: {ctx_source}")
    print(f"Max scenes (fast): {args.llm_fast_max_scenes}")
    print(f"LLM pipeline: {args.llm_pipeline}")
    print(f"Force deep on all: {args.llm_force_deep}")

    # Select scenes to analyze (same logic as analyzer)
    if len(scenes) > args.llm_fast_max_scenes:
        sorted_by_q = sorted(range(len(scenes)),
                             key=lambda i: -scenes[i].get("quality_score", 0))
        top_candidates = sorted_by_q[:args.llm_fast_max_scenes * 2]
        top_candidates.sort(key=lambda i: scenes[i].get("best_frame", 0))
        step = max(1, len(top_candidates) // args.llm_fast_max_scenes)
        selected = top_candidates[::step][:args.llm_fast_max_scenes]
        analyze_set = set(selected)
        print(f"Selected {len(analyze_set)} scenes from {len(scenes)}")
    else:
        analyze_set = set(range(len(scenes)))
        print(f"Analyzing all {len(scenes)} scenes")

    # ── Fast model pass ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FAST MODEL: {args.llm_fast_model}")
    print(f"{'='*60}")

    # Collect scenes to analyze
    fast_scenes = [(si, scene) for si, scene in enumerate(scenes) if si in analyze_set]

    # Build task list
    task_list = []
    task_meta = []  # (si, vlabel, vpath)
    for si, scene in fast_scenes:
        if args.llm_pipeline == "max":
            variants = find_all_variants(scenes_dir, scene)
        else:
            path = find_scene_image(scenes_dir, scene)
            variants = [("v3_aggressive_shadow", path)] if path else []
        if not variants:
            print(f"  [{si:3d}] f{scene['best_frame']:5d} MISSING image")
            continue
        for vlabel, vpath in variants:
             # Fast pass is ALWAYS single-stage (two-stage is deep-pass only).
            task_list.append(lambda vpath=vpath: llm_analyze(
                args.llm_fast_model, vpath, prompt))
            task_meta.append((si, vlabel, vpath))

    t0 = time.time()
    def _on_complete(idx, _result):
        si, vlabel, vpath = task_meta[idx]
        scene = scenes[si]
        result, elapsed, err = _result
        if err:
            print(f"    [{si:3d}] f{scene['best_frame']:5d} {vlabel} ERROR: {err[:100]}")
        else:
            v_found = result.get("objects_found", False)
            v_conf = result.get("confidence", 0)
            v_findings = result.get("findings", [])
            print(f"    [{si:3d}] f{scene['best_frame']:5d} {vlabel}: "
                  f"found={v_found} conf={v_conf} findings={len(v_findings)} "
                  f"({elapsed:.1f}s)")

    results = parallel_llm_batch(task_list, workers=args.llm_parallel, on_complete=_on_complete)
    elapsed_total = time.time() - t0

    # Merge results back into scenes
    scene_variants = {}  # si -> [(vlabel, vpath, result_tuple)]
    for idx, (si, vlabel, vpath) in enumerate(task_meta):
        scene_variants.setdefault(si, []).append((vlabel, vpath, results[idx]))

    analyzed = 0
    for si, scene in fast_scenes:
        variant_data = scene_variants.get(si, [])
        if not variant_data:
            continue

        merged_findings = []
        merged_found = False
        merged_conf = 0
        merged_terrain = ""
        variant_results = {}

        for vlabel, vpath, (result, elapsed, err) in variant_data:
            if err:
                variant_results[vlabel] = {"error": err[:500]}
                continue
            variant_results[vlabel] = result
            v_found = result.get("objects_found", False)
            v_conf = result.get("confidence", 0)
            v_findings = result.get("findings", [])
            if v_found:
                merged_found = True
                merged_conf = max(merged_conf, v_conf)
                for fnd in v_findings:
                    fnd["source_variant"] = vlabel
                    merged_findings.append(fnd)
                    print(f"         -> {fnd.get('type','?')} {fnd.get('zone','')} "
                          f"color={fnd.get('color','?')} "
                          f"{fnd.get('description','')[:100]}")
            merged_terrain = result.get("terrain", merged_terrain)

        scene["llm_fast"] = {
            "objects_found": merged_found,
            "confidence": merged_conf,
            "description": variant_results.get("v3_aggressive_shadow",
                              variant_results.get("v1_high_contrast", {})).get("description", ""),
            "findings": merged_findings,
            "terrain": merged_terrain,
            "variant_results": variant_results,
        }
        scene["llm_fast"]["llm_time"] = sum(
            v.get("llm_time", 0) for v in variant_results.values()
            if isinstance(v, dict))
        scene["llm_fast"]["llm_model"] = args.llm_fast_model

        analyzed += 1
        if analyzed % 10 == 0:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFast model done: {analyzed} scenes in {elapsed_total:.0f}s")

    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Deep model on positives (or all if --llm-force-deep) ──────────
    # Heuristic scorer — ranks scenes by deep LLM value
    TARGET_COLORS = {"orange", "bright orange", "reddish-orange", "blue", "dark blue",
                     "blue-grey", "black", "dark grey", "dark gray", "dark brown"}

    def deep_priority_score(scene):
        """Heuristic score for deep LLM budget allocation."""
        score = 0
        llm = scene.get("llm_fast", {})
        if not isinstance(llm, dict):
            return 0
        findings = llm.get("findings", [])
        score += len(findings) * 2

        llm_zones = set()
        for f in findings:
            color = (f.get("color", "") or "").lower()
            if any(tc in color for tc in TARGET_COLORS):
                score += 5
            score += f.get("confidence", 0) * 3
            zone = f.get("zone", "")
            if zone:
                llm_zones.add(zone)

        # Color anomaly overlap
        dynamics = scene.get("dynamics", {})
        if isinstance(dynamics, dict):
            ca_findings = dynamics.get("color_findings", [])
            if ca_findings:
                ca_zones = {f"Z{f.get('zone', '')}" for f in ca_findings}
                overlap = llm_zones & ca_zones
                score += len(overlap) * 4

        return score

    if args.llm_force_deep:
        deep_targets = [scenes[i] for i in sorted(analyze_set)
                        if find_scene_image(scenes_dir, scenes[i])]
    else:
        positives = [s for s in scenes
                     if s.get("llm_fast", {}).get("objects_found", False)
                     or s.get("llm_fast", {}).get("confidence", 0) > 0.4]
        if args.llm_pipeline == "chancepeek":
            negatives = [s for s in scenes
                        if not s.get("llm_fast", {}).get("objects_found", False)
                        and s.get("llm_fast", {}).get("confidence", 0) <= 0.4]
            candidate_pool = positives + negatives
        else:
            candidate_pool = positives

        # Rank by heuristic priority score
        if len(candidate_pool) > args.llm_deep_max_scenes:
            scored = [(deep_priority_score(s), s) for s in candidate_pool]
            scored.sort(key=lambda x: -x[0])
            deep_targets = [s for _, s in scored[:args.llm_deep_max_scenes]]
            deep_targets.sort(key=lambda s: s.get("best_frame", 0))
            print(f"\n  Deep budget triage: {len(candidate_pool)} candidates → top {len(deep_targets)} by priority score")
            print(f"  Top 5 scores: {[(s.get('best_frame', 0), round(sc, 1)) for sc, s in scored[:5]]}")
            print(f"  Cut-off score: {round(scored[args.llm_deep_max_scenes-1][0], 1)} (scene f{scored[args.llm_deep_max_scenes-1][1].get('best_frame', 0)})")
        else:
            deep_targets = candidate_pool

    if deep_targets and args.llm_deep_model:
        print(f"\n{'='*60}")
        if args.llm_force_deep:
            label = "ALL scenes"
        elif args.llm_pipeline == "chancepeek":
            label = f"{len(deep_targets)} scenes (positives + chancepeek)"
        else:
            label = f"{len(deep_targets)} positive scenes"
        stage_tag = " [TWO-STAGE]" if not args.llm_no_two_stage else ""
        print(f"DEEP MODEL: {args.llm_deep_model} ({label}){stage_tag}")
        print(f"{'='*60}")

        # Build deep task list
        deep_task_list = []
        deep_task_meta = []  # (scene, path)
        for scene in deep_targets:
            path = find_scene_image(scenes_dir, scene)
            if not path:
                continue
            if not args.llm_no_two_stage:
                deep_task_list.append(lambda path=path: two_stage_analyze(
                    args.llm_deep_model, args.llm_reasoning_model, path,
                    mission_context=mission_context, timeout_vision=180))
            else:
                deep_task_list.append(lambda path=path: llm_analyze(
                    args.llm_deep_model, path, prompt, timeout=180))
            deep_task_meta.append((scene, path))

        t_deep = time.time()

        # Live progress per completed task (mirrors az-video), even sequential.
        def _on_deep_complete(idx, _result):
            scene, path = deep_task_meta[idx]
            result, elapsed, err = _result
            if err:
                print(f"  f{scene['best_frame']:5d} ERROR: {err[:200]}")
            else:
                found = result.get("objects_found", False)
                conf = result.get("confidence", 0)
                print(f"  f{scene['best_frame']:5d} t={scene.get('time_str','')} "
                      f"found={found} conf={conf} ({elapsed:.1f}s)")

        deep_results = parallel_llm_batch(deep_task_list, workers=args.llm_parallel,
                                          on_complete=_on_deep_complete)
        t_deep_total = time.time() - t_deep

        for idx, (scene, path) in enumerate(deep_task_meta):
            result, elapsed, err = deep_results[idx]
            if err:
                scene["llm_deep"] = {"error": err[:500]}
            else:
                scene["llm_deep"] = result
                if result.get("objects_found", False):
                    for fnd in result.get("findings", []):
                        print(f"      -> {fnd.get('type','?')} {fnd.get('zone','')} "
                              f"color={fnd.get('color','?')} "
                              f"{fnd.get('description','')[:100]}")

        print(f"\n  Deep pass done: {len(deep_task_meta)} scenes in {t_deep_total:.0f}s")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────
    positives = [s for s in scenes
                 if s.get("llm_fast", {}).get("objects_found", False)]
    deep_positives = [s for s in scenes
                      if s.get("llm_deep", {}).get("objects_found", False)]
    print(f"\n{'='*60}")
    print("RE-RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Scenes analyzed: {analyzed}")
    print(f"Fast positives: {len(positives)}")
    print(f"Deep confirmed: {len(deep_positives)}")
    for s in positives:
        llm = s.get("llm_fast", {})
        print(f"  f{s['best_frame']} t={s.get('time_str','')} conf={llm.get('confidence',0)}")
        for fnd in llm.get("findings", []):
            print(f"    -> {fnd.get('type','?')} {fnd.get('zone','')} "
                  f"color={fnd.get('color','?')} {fnd.get('description','')[:100]}")
    for s in deep_positives:
        llm = s.get("llm_deep", {})
        print(f"  DEEP: f{s['best_frame']} t={s.get('time_str','')} conf={llm.get('confidence',0)}")
        for fnd in llm.get("findings", []):
            print(f"    -> {fnd.get('type','?')} {fnd.get('zone','')} "
                  f"color={fnd.get('color','?')} {fnd.get('description','')[:100]}")


if __name__ == "__main__":
    main()
