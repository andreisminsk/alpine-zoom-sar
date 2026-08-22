# az-geometry — Improvement Notes

Review of `src/alpine_zoom/geometry.py` (695 lines).

---

## Recognition Capability Boosts (primary focus)

The SAR targets are **rope, tent, bivouac, fixed gear, person** — all *small, thin, or
low-contrast* on snow. The current pipeline (single Canny → `RETR_EXTERNAL` contours →
Hough) is tuned for *large, high-contrast, closed* shapes. It structurally misses the
things that matter most.

### 1. Thin / elongated shape detector (rope) — highest SAR value
The #1 target. A rope is a long, thin, possibly-curved line. Today it only survives as
fragmented Hough segments. Add:
- Elongated-contour detection (high aspect-ratio, low area/perimeter ratio)
- **Segment chaining** to join collinear Hough fragments into one continuous rope
- Arc/curve fitting for "gently curved" ropes (not just straight Hough lines)

### 2. Multi-scale pyramid
Small objects (helmet, crampons, a person ~10–15px) die at the current scale.
Run detection at 2–3 scales (e.g. 1×, 2×, 4× downsampled) and merge results.

### 3. Adaptive edge thresholding
Fixed Canny 50/150 fails on low-contrast snow (dark rope on snow).
Use Otsu or per-region adaptive thresholds so edges survive in flat, low-contrast areas.

### 4. `RETR_CCOMP` instead of `RETR_EXTERNAL`
Catches nested shapes (pole inside tent, gear inside bivouac, rope through anchor).

### 5. Corner-density map
Man-made objects have many sharp corners; natural terrain has few.
A local Shi-Tomasi / Harris corner density is a strong "this is synthetic" signal —
currently absent. High corner density in a small region → flag.

### 6. Color + geometry fusion
A blue/orange region that is *also* geometrically regular is a near-certain finding.
Fuse `color.py` and `geometry.py` outputs instead of running them blind.
Cross-reference: color finding in zone Z3 + geometry finding in zone Z3 → boost confidence.

### 7. Curve / arc detection
Ropes are "gently curved" — add arc fitting (e.g. `cv2.fitArcEllipse` on contour
points) alongside straight Hough lines.

---

## Maintenance / Cleanup (secondary)

### Performance
- **`verify_circle_gradient` is O(N_circles × image).** Each call recomputes
  `GaussianBlur` + two `Sobel` + full `Canny`. Precompute once, pass in.
- **Inconsistent blur.** `HoughCircles` runs on unblurred `gray`;
  `verify_circle_gradient` re-blurs internally. Blur once, share.
- **Zone-assignment loop duplicated 3×.** Extract `find_zone(cx, cy, zones)`.
- **`zone_grid` / `ignore_mask` duplicated** across `common.py`, `color.py`, `geometry.py`.
  Move to `common.py`.

### Correctness
- **`deduplicate` discards the loser entirely** (unlike `color.py` which merges area).
  Consider union-bbox / max-confidence merge.
- **`draw_findings` `shape_colors` has bogus `"magenta"` key** and no `polygon-N` handling.
- **`line_similarity` overlap math is hand-wavy** → false merges of unrelated parallels.
- **`side_ratio = min(sides) / max(sides + [1e-10])`** — guards wrong side.

### Consistency
- **Docstring vs CLI mismatch:** docstring advertises single-image mode; `main()` only
  accepts `output_dir` with `report.json`.
- **No report integration.** `color.py` writes findings via `build_dynamics` +
  `--update-report`; geometry is standalone-only.
- **`bbox_iou` / `line_similarity` defined after `deduplicate`** uses them.

---

## Suggested Implementation Order

| Phase | Items | Rationale |
|-------|-------|-----------|
| 1 | #1 (rope), #3 (adaptive edges) | Highest SAR payoff, moderate effort |
| 2 | #2 (multi-scale), #4 (RETR_CCOMP) | Catches small/nested objects |
| 3 | #5 (corner density), #7 (arcs) | New signal types |
| 4 | #6 (color+geometry fusion) | Requires cross-module integration |
| 5 | Maintenance items | Low risk, do alongside |
