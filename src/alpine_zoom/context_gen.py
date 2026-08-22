"""Generate a mission context JSON file from a free-text situation description.

Uses a reasoning LLM (default: glm-5.1:cloud) to turn a natural-language
description of a search scenario into a structured MissionContext JSON,
compatible with the contexts/*.json files consumed by the SAR pipeline.

Usage:
  az-context -i "description text" -o contexts/my-mission.json
  az-context -i description.txt -o contexts/my-mission.json
  az-context -i desc.txt -o out.json --llm-reasoning-model qwen3.5:397b-cloud
"""
import sys
import os
import json
import argparse

sys.stdout.reconfigure(encoding="utf-8")

from alpine_zoom.llm import llm_text_analyze
from alpine_zoom.context import MissionContext

DEFAULT_REASONING_MODEL = "glm-5.1:cloud"

# Fields that make up a MissionContext (mirrors contexts/*.json schema).
CONTEXT_FIELDS = (
    "context",
    "natural_list",
    "priority_signals",
    "target_objects",
    "platform_notes",
    "color_shift_notes",
)

GEN_PROMPT = """You are a search & rescue mission planner. Given a free-text \
description of a search situation, produce a structured MISSION CONTEXT that \
will be appended to a vision-analysis prompt used to scan drone/helicopter \
footage for lost people or signs of catastrophe.

The output must be a JSON object with EXACTLY these fields (all strings, \
use "" for fields that do not apply — do not omit them):

{
  "context": "One-paragraph summary of the mission: environment, altitude, \
what is being searched for, and the key visual cues to watch for.",
  "natural_list": "Comma-separated list of NATURAL elements that are expected \
and should NOT be flagged (terrain, minerals, shadows, weather, etc.).",
  "priority_signals": "Numbered list of the highest-priority visual signals to \
look for, most important first. For each, note how to distinguish it from \
natural look-alikes. When in doubt, report.",
  "target_objects": "List of man-made objects or signs to look for (people, \
gear, structures, marks), with brief visual descriptors for each.",
  "platform_notes": "Notes about the capture platform (drone, helicopter, \
ground) and any elements to IGNORE (e.g. aircraft interior). Use \"\" if \
not applicable.",
  "color_shift_notes": "Notes on how lighting/shadow can shift colors so the \
analyst does not rely on color alone. Use \"\" if not applicable."
}

Rules:
- Be specific and actionable; the analyst is a vision model scanning small \
regions of terrain.
- Distinguish priority signals from natural look-alikes explicitly.
- Prefer reporting false alarms over missing real findings.
- Output JSON ONLY. No prose, no markdown fences.

SITUATION DESCRIPTION:
__DESCRIPTION__
"""


def read_input(input_arg):
    """Return the situation text. If input_arg is an existing file, read it;
    otherwise treat it as literal text."""
    if os.path.isfile(input_arg):
        with open(input_arg, encoding="utf-8") as f:
            return f.read()
    return input_arg


def generate_context(description, model=DEFAULT_REASONING_MODEL, timeout=180):
    """Call the reasoning LLM to produce a MissionContext from a description.

    Returns (MissionContext, elapsed_sec, error_str_or_None).
    """
    prompt = GEN_PROMPT.replace("__DESCRIPTION__", description.strip())
    result, elapsed, error = llm_text_analyze(model, prompt, timeout=timeout)
    if error:
        return None, elapsed, error
    if not result:
        return None, elapsed, "LLM returned no result"

    # Extract the context fields; ignore any extra LLM-added keys.
    data = {k: result.get(k, "") for k in CONTEXT_FIELDS}
    # Ensure all are strings.
    for k in CONTEXT_FIELDS:
        if not isinstance(data[k], str):
            data[k] = "" if data[k] is None else str(data[k])

    # Derive a name from the output path later; leave blank here.
    ctx = MissionContext(**data)
    return ctx, elapsed, None


def main():
    p = argparse.ArgumentParser(
        description="Generate a mission context JSON from a text description.")
    p.add_argument("-i", "--input", required=True,
                   help="Situation description: literal text or path to a text file.")
    p.add_argument("-o", "--output", required=True,
                   help="Output JSON path (e.g. contexts/my-mission.json).")
    p.add_argument("--llm-reasoning-model", default=DEFAULT_REASONING_MODEL,
                   dest="llm_reasoning_model",
                   help=f"Reasoning LLM model (default: {DEFAULT_REASONING_MODEL}).")
    p.add_argument("--llm-timeout", type=int, default=180,
                   help="LLM timeout in seconds (default: 180).")
    p.add_argument("--name", default=None,
                   help="Preset name for the context (default: output filename stem).")
    args = p.parse_args()

    description = read_input(args.input)
    if not description.strip():
        print("ERROR: empty input description.")
        sys.exit(1)

    print(f"Reasoning model: {args.llm_reasoning_model}")
    print(f"Input: {args.input} ({len(description)} chars)")
    print("Generating mission context...")

    ctx, elapsed, error = generate_context(
        description, model=args.llm_reasoning_model, timeout=args.llm_timeout)

    if error:
        print(f"ERROR: {error} ({elapsed:.1f}s)")
        sys.exit(1)

    # Set the name: explicit --name, else output filename stem.
    if args.name:
        ctx.name = args.name
    else:
        ctx.name = os.path.splitext(os.path.basename(args.output))[0]

    # Write output JSON (pretty, matching contexts/*.json style).
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ctx.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {args.output} ({elapsed:.1f}s)")
    print(f"  name: {ctx.name}")
    for k in CONTEXT_FIELDS:
        val = getattr(ctx, k)
        preview = val.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  {k}: {preview if preview else '(empty)'}")


if __name__ == "__main__":
    main()
