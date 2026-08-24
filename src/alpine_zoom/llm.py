"""LLM client + prompt building for SAR video analysis.

Unified LLM interface — all derivatives import from here instead of
duplicating the Ollama HTTP client and prompt logic.

Extracted from alpine_zoom.video (source of truth).
"""
import sys
import os
import json
import time
import base64
import urllib.request
import concurrent.futures

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.context import (
    FOUNDATIONAL_PROMPT,
    MissionContext,
    build_prompt as _build_prompt,
    get_context,
    default_sar_context,
    default_sar_heli_context,
    VISION_PROMPT,
    build_reasoning_prompt,
)

# Backward-compatible alias (foundational prompt only)
SAR_PROMPT = FOUNDATIONAL_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"


# ── Prompt building ────────────────────────────────────────────────────

def build_prompt(helicopter=False, mission_context=None):
    """Build the full LLM prompt.

    If mission_context is provided, it is appended to the foundational prompt.
    If mission_context is None and helicopter=True, the SAR helicopter preset
    is loaded automatically (backward compatibility).
    If both are None, only the foundational prompt is returned (generic mode).
    """
    if mission_context is None and helicopter:
        mission_context = default_sar_heli_context()
    return _build_prompt(mission_context)


def resolve_mission_context(context_file=None, context_preset=None,
                            helicopter=False, report=None):
    """Resolve a MissionContext from file, preset, report, or helicopter flag.

    Priority: context_file > context_preset > stored in report > helicopter > none.

    Returns (MissionContext or None, source_description_str).
    """
    if context_file:
        ctx = get_context(context_file=context_file)
        return ctx, f"file: {context_file}"
    if context_preset:
        ctx = get_context(preset=context_preset)
        return ctx, f"preset: {context_preset}"

    # Try stored context in report
    if report:
        stored = report.get("analysis_params", {}).get("mission_context")
        if stored and any(stored.get(k) for k in
                ("context", "natural_list", "priority_signals",
                 "target_objects", "platform_notes", "color_shift_notes")):
            ctx = MissionContext(**stored)
            return ctx, f"stored in report.json: {stored.get('name', 'custom')}"

    if helicopter:
        ctx = get_context(helicopter=True)
        return ctx, "helicopter flag (sar-heli preset)"

    ctx = get_context()  # default_sar=True → loads 'sar' preset
    return ctx, "default (sar preset)"


# ── Image encoding ───────────────────────────────────────────────────

def encode_image(path):
    """Read image file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── LLM analysis ──────────────────────────────────────────────────────

def llm_analyze(model, image_path, prompt=None, timeout=120):
    """Send image to vision LLM, return parsed result.

    Args:
        model: Ollama model name
        image_path: path to image file
        prompt: LLM prompt string (default: SAR_PROMPT / foundational prompt)
        timeout: request timeout in seconds

    Returns: (result_dict, elapsed_sec, error_str_or_None)
    """
    if prompt is None:
        prompt = SAR_PROMPT
    b64 = encode_image(image_path)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        content = result.get("message", {}).get("content", "")
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            parsed["llm_time"] = round(elapsed, 1)
            parsed["llm_model"] = model
            return parsed, elapsed, None
        return {"raw": content[:3000], "llm_time": round(elapsed, 1),
                "llm_model": model, "objects_found": False, "confidence": 0}, elapsed, None
    except Exception as e:
        return None, time.time() - t0, str(e)


def llm_text_analyze(model, prompt, timeout=120):
    """Send text-only prompt to LLM (no image), return parsed JSON result.
    Used for Stage 2 reasoning in two-stage mode."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        content = result.get("message", {}).get("content", "")
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            parsed["llm_time"] = round(elapsed, 1)
            parsed["llm_model"] = model
            return parsed, elapsed, None
        return {"raw": content[:3000], "llm_time": round(elapsed, 1),
                "llm_model": model, "objects_found": False, "confidence": 0}, elapsed, None
    except Exception as e:
        return None, time.time() - t0, str(e)


def two_stage_analyze(vision_model, reasoning_model, image_path, mission_context=None,
                      timeout_vision=180, timeout_reasoning=120):
    """Two-stage analysis: vision model describes, reasoning model concludes.

    Stage 1: vision model (e.g. qwen3.5:397b-cloud) sees image, outputs structured description
    Stage 2: reasoning model (e.g. glm-5.1:cloud) takes description + mission context, concludes

    Returns (result_dict, total_elapsed, error).
    result_dict has same schema as llm_analyze output for compatibility.
    """
    t0 = time.time()

    # Stage 1: Vision description
    b64 = encode_image(image_path)
    vision_payload = {
        "model": vision_model,
        "messages": [{"role": "user", "content": VISION_PROMPT, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    data = json.dumps(vision_payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_vision) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        vision_elapsed = time.time() - t0
        content = result.get("message", {}).get("content", "")
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            return ({"raw": content[:3000], "objects_found": False, "confidence": 0,
                     "llm_time": round(vision_elapsed, 1), "llm_model": vision_model},
                    vision_elapsed, "vision stage: no JSON in response")
        vision_desc = json.loads(content[start:end])
    except Exception as e:
        return None, time.time() - t0, f"vision stage: {e}"

    # Stage 2: Reasoning from description
    reasoning_prompt = build_reasoning_prompt(vision_desc, mission_context)
    reasoning_result, reasoning_elapsed, reasoning_err = llm_text_analyze(
        reasoning_model, reasoning_prompt, timeout=timeout_reasoning)

    total_elapsed = time.time() - t0

    if reasoning_err:
        return ({"objects_found": False, "confidence": 0,
                 "description": "vision stage OK, reasoning stage failed",
                 "vision_description": vision_desc,
                 "findings": [],
                 "llm_time": round(total_elapsed, 1),
                 "llm_model": f"{vision_model}+{reasoning_model}",
                 "reasoning_error": reasoning_err},
                total_elapsed, reasoning_err)

    reasoning_result["vision_description"] = vision_desc
    reasoning_result["llm_time"] = round(total_elapsed, 1)
    reasoning_result["llm_model"] = f"{vision_model}+{reasoning_model}"
    return reasoning_result, total_elapsed, None


# ── Variant selection ─────────────────────────────────────────────────

def get_llm_variants(scene, full=False):
    """Get list of (label, path) variants to analyze.

    Uses in-memory scene dict keys (image_path_vN, orig_path).
    For report.json-based scenes, use alpine_zoom.common.find_all_variants() instead.
    """
    if full:
        variants = []
        if scene.get("orig_path"):
            variants.append(("orig", scene["orig_path"]))
        if scene.get("image_path_v1"):
            variants.append(("v1_high_contrast", scene["image_path_v1"]))
        if scene.get("image_path_v2"):
            variants.append(("v2_gentle_clahe", scene["image_path_v2"]))
        if scene.get("image_path_v3"):
            variants.append(("v3_aggressive_shadow", scene["image_path_v3"]))
        if variants:
            return variants
    # Default: v3 only
    if scene.get("image_path_v3"):
        return [("v3_aggressive_shadow", scene["image_path_v3"])]
    if scene.get("image_path"):
        return [("default", scene["image_path"])]
    return []


# ── Parallel batch execution ──────────────────────────────────────────

def parallel_llm_batch(tasks, workers=4, on_complete=None):
    """Run a batch of LLM analysis tasks in parallel.

    Args:
        tasks: list of callables (no args) that return (result, elapsed, error)
        workers: number of parallel workers (0 or 1 = sequential)
        on_complete: optional callback(idx, result) invoked as each task finishes,
            giving live progress even in sequential mode (mirrors az-video).

    Returns:
        list of (result, elapsed, error) in the same order as input tasks
    """
    if workers <= 1 or len(tasks) <= 1:
        results = []
        for i, task in enumerate(tasks):
            r = task()
            if on_complete:
                on_complete(i, r)
            results.append(r)
        return results

    results = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task): i for i, task in enumerate(tasks)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            if on_complete:
                on_complete(idx, results[idx])
    return results
