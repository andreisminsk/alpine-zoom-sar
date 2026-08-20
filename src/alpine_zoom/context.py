"""Mission context configuration for the two-part prompt system.

The LLM prompt is split into:
  1. FOUNDATIONAL_PROMPT — built-in, generic anomaly detection (not editable)
  2. MissionContext — user-configurable per-mission context

Context can be loaded from a JSON file, a named preset, or built inline.
"""
import json
import os
from dataclasses import dataclass, field, asdict


# ── Foundational prompt (built-in, not editable) ──────────────────────

FOUNDATIONAL_PROMPT = """You are a vision analyst for remote scene inspection. Your task: examine \
the image and identify anything that does NOT belong in the scene — \
objects, colors, shapes, or textures that break the natural pattern of \
the environment.

You are the last line of analysis. Other systems have already processed \
this image and found nothing. If you also miss something important, the \
consequences may be severe.

STRATEGY: Scan zones in order Z1 through Z9. For each zone, note anything \
unusual in one line, then move to the next zone. Do not revisit zones. \
Respond with your first impression — do not reconsider, revise, or \
second-guess. Your first answer is your best answer. Output your first \
conclusion and do not revise it.

FAST EXIT: If after scanning Z1-Z3 you see only natural terrain with no \
unusual colors, shapes, or anomalies, you may output objects_found=false \
immediately without scanning Z4-Z9.

If after scanning all zones you see nothing unusual, output objects_found=false \
with empty findings. Do not describe normal terrain.

Do not plan your JSON output in your reasoning — just output it directly.

ANALYZE ALONG TWO AXES:

1) COLORS distinct from the scene:
   Any small area whose color is unusual for the environment. Even a
   faint tint in a small region can be significant.

2) GEOMETRICAL SHAPES distinct from the scene:
   Straight lines, right angles, circles, polygons, thin extended
   lines — shapes that nature rarely produces.

When in doubt, report. It is far better to report a false alarm than to \
miss a real finding.

CONFIDENCE CALIBRATION: If you find a clearly synthetic object (unnatural \
color AND geometric shape), confidence should be 0.9+. If you find an \
unusual color OR shape (but not both), confidence 0.6-0.8. If you find \
a faint or ambiguous signal, confidence 0.3-0.5.

IGNORE camera overlays: the 3x3 zone grid (Z1 top-left through Z9
bottom-right), filename text, timestamps, GPS coordinates, and
telemetry text burned into the image. These are not part of the scene.

Use zone numbers (Z1-Z9) to locate findings.

Answer in JSON ONLY:
{
  "objects_found": true/false,
  "confidence": 0.0-1.0,
  "findings": [
    {"type": "object/anomaly/color/shape/mark",
     "zone": "Z1-Z9",
     "color": "color name or 'unknown'",
     "man_made": "likely/possible/unlikely/unknown",
     "confidence": 0.0-1.0,
     "description": "what you see and why it stands out"}
  ],
  "terrain": "environment type",
  "visibility": "good/moderate/poor/obscured",
  "zones_of_interest": ["Z1", "Z3"]
}"""


# ── Mission context ───────────────────────────────────────────────────

@dataclass
class MissionContext:
    """User-configurable context appended to the foundational prompt."""
    context: str = ""
    natural_list: str = ""
    priority_signals: str = ""
    target_objects: str = ""
    platform_notes: str = ""
    color_shift_notes: str = ""
    name: str = ""  # preset name, for traceability

    def to_prompt(self):
        """Serialize non-empty fields into the ==== MISSION CONTEXT ==== block."""
        sections = []
        if self.context:
            sections.append(f"CONTEXT: {self.context}")
        if self.natural_list:
            sections.append(f"NATURAL LIST: {self.natural_list}")
        if self.priority_signals:
            sections.append(f"PRIORITY SIGNALS: {self.priority_signals}")
        if self.target_objects:
            sections.append(f"TARGET OBJECTS: {self.target_objects}")
        if self.platform_notes:
            sections.append(f"PLATFORM NOTES: {self.platform_notes}")
        if self.color_shift_notes:
            sections.append(f"COLOR SHIFT NOTES: {self.color_shift_notes}")
        if not sections:
            return ""
        return "\n\n==== MISSION CONTEXT ====\n\n" + "\n\n".join(sections)

    def to_dict(self):
        return asdict(self)


# ── Presets ───────────────────────────────────────────────────────────

def _sar_base_context():
    """SAR context without helicopter notes (shared by sar and sar-heli)."""
    return MissionContext(
        name="sar",
        context=(
            "High mountains above 4000 meters. Looking for lost climbers, "
            "signs of their presence, signs of catastrophe. Attention to "
            "blue, orange wear or gear, signs of gear use or loss."
        ),
        natural_list=(
            "Ice, snow, boulders, cracks, stones, rock formations, minerals, "
            "lichen on rocks, red water algae on snow, iron-rich rocks "
            "(reddish/orange), shadows from rock formations, falling snow."
        ),
        priority_signals=(
            "1. ORANGE SPOT — small discrete bright orange on rock/snow/scree, "
            "top priority. Distinguish from lichen (flat/matted on rock) and "
            "iron-rich soil (large diffuse areas). When in doubt, report.\n"
            "2. BLUE SPOT — almost never natural on mountain terrain. Report "
            "even a faint blue tint in a small area.\n"
            "3. DARK SPOT on bright snow — the strongest signal overall. "
            "Any dark shape on bright snow that is not a shadow or rock."
        ),
        target_objects=(
            "Person (small dark/colored shape on snow, could be partially "
            "buried). Rope (thin straight or gently curved line — mountains "
            "do not produce thin straight lines; look along fall lines, "
            "between rocks, at snow patch edges). Tent (polygon shape). "
            "Ice axe (thin line with pick). Crampons, backpack, helmet, "
            "sleeping bag. Bivouac (dug platform, snow cave entrance, "
            "ventilation hole). Fall marks (groove in snow, impact crater, "
            "scattered gear trail). Fixed gear (slings/loops on rocks, "
            "pitons, anchors, fixed rope on rock faces)."
        ),
        color_shift_notes=(
            "In shadow on snow, colors shift — a red jacket looks dark gray, "
            "blue can look dark. Do NOT rely on color alone. Look for shapes, "
            "geometry, shadows, and 'anything that is NOT snow and NOT rock'."
        ),
    )


def default_sar_context():
    """SAR preset for drone footage."""
    return _sar_base_context()


def default_sar_heli_context():
    """SAR preset for helicopter footage — adds platform notes."""
    ctx = _sar_base_context()
    ctx.name = "sar-heli"
    ctx.platform_notes = (
        "Helicopter footage — some frames show helicopter interior: "
        "passengers, seats, window frames, door edges, seatbelts, headset "
        "cables, reflections in glass. IGNORE all interior elements. ONLY "
        "look for climbers and gear ON THE TERRAIN OUTSIDE the helicopter. "
        "If part of the frame shows interior and part shows exterior, focus "
        "ONLY on exterior terrain. Passengers inside the aircraft are NOT "
        "the missing climbers — the missing climbers are on the mountain, "
        "visible from above as small shapes on snow or rock."
    )
    return ctx


PRESETS = {
    "sar": default_sar_context,
    "sar-heli": default_sar_heli_context,
}


def load_context(path):
    """Load a MissionContext from a JSON file.

    Expected keys: context, natural_list, priority_signals,
    target_objects, platform_notes, color_shift_notes, name (all optional).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return MissionContext(
        context=data.get("context", ""),
        natural_list=data.get("natural_list", ""),
        priority_signals=data.get("priority_signals", ""),
        target_objects=data.get("target_objects", ""),
        platform_notes=data.get("platform_notes", ""),
        color_shift_notes=data.get("color_shift_notes", ""),
        name=data.get("name", os.path.splitext(os.path.basename(path))[0]),
    )


def get_context(preset=None, context_file=None, helicopter=False):
    """Resolve a MissionContext from preset name, file path, or helicopter flag.

    Priority: context_file > preset > helicopter flag > None.
    """
    if context_file:
        return load_context(context_file)
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
        return PRESETS[preset]()
    if helicopter:
        return default_sar_heli_context()
    return None


def build_prompt(mission_context=None):
    """Build the full prompt: foundational + optional mission context."""
    prompt = FOUNDATIONAL_PROMPT
    if mission_context:
        prompt += mission_context.to_prompt()
    return prompt


# ── Two-stage prompts ─────────────────────────────────────────────────

VISION_PROMPT = """You are a vision analyst. Describe what you see in this image. \
Do NOT judge, do NOT reason, do NOT decide what is natural or man-made. \
Your only job is to PERCEIVE and LIST what is visible.

Scan zones in order Z1 through Z9. For each zone, list every distinct \
element you see — colors, shapes, textures, lines, spots, shadows. \
Be thorough but concise: one line per element.

Pay special attention to:
- Small colored spots (note exact color, approximate size, zone)
- Thin lines or straight edges (note direction, zone)
- Geometric shapes (rectangles, circles, polygons — note zone)
- Bright glints or reflections (note zone)
- Dark spots on light surfaces (note zone, size)
- Anything that has a clear edge or boundary

IGNORE camera overlays: zone grid lines, filename text, timestamps, \
GPS coordinates, telemetry text. These are not part of the scene.

Answer in JSON ONLY:
{
  "elements": [
    {"zone": "Z1-Z9",
     "type": "color_spot/line/shape/glint/dark_spot/texture/other",
     "color": "color name or 'none'",
     "size": "small/medium/large",
     "description": "one-line factual description of what you see"}
  ],
  "terrain": "environment type",
  "visibility": "good/moderate/poor/obscured"
}

Do NOT include a "findings" or "objects_found" field. Do NOT judge \
whether things belong or not. Just list what you see."""


def build_reasoning_prompt(description_json, mission_context=None):
    """Build the Stage 2 reasoning prompt from a vision description + context.

    Args:
        description_json: JSON string or dict from Stage 1 (vision model)
        mission_context: optional MissionContext
    """
    if isinstance(description_json, (dict, list)):
        import json as _json
        desc_str = _json.dumps(description_json, indent=2, ensure_ascii=False)
    else:
        desc_str = str(description_json)

    prompt = f"""You are a reasoning analyst. A vision model has described the \
contents of a scene image. Your job is to analyze that description and \
decide whether anything does NOT belong in the scene.

Apply the mission context rules below to the scene description. Flag \
anything that matches priority signals, target objects, or is not in \
the natural list. When in doubt, report — it is far better to report a \
false alarm than to miss a real finding.

Work fast. Do not overthink. Apply the rules, output your conclusion.

SCENE DESCRIPTION FROM VISION ANALYST:
{desc_str}
"""
    if mission_context:
        prompt += mission_context.to_prompt()

    prompt += """

Answer in JSON ONLY:
{
  "objects_found": true/false,
  "confidence": 0.0-1.0,
  "findings": [
    {"type": "object/anomaly/color/shape/mark",
     "zone": "Z1-Z9",
     "color": "color name or 'unknown'",
     "man_made": "likely/possible/unlikely/unknown",
     "confidence": 0.0-1.0,
     "description": "what it is and why it does not belong"}
  ],
  "zones_of_interest": ["Z1", "Z3"],
  "notes": "brief reasoning summary"
}"""
    return prompt
