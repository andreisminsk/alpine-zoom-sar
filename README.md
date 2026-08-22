# Alpine Zoom — AI-Assisted Mountain Search & Rescue Video Analysis

A toolset for analyzing drone and helicopter video footage to find lost climbers 
in mountain terrain. Uses a hybrid pipeline: classical computer vision for frame 
quality filtering, scene grouping, and color anomaly detection, then
vision-capable LLMs for visual analysis.

## Setup & Installation

### 1. Python dependencies

```bash
pip install -e .
```

This installs 10 console commands on your PATH:

| Command | Purpose |
|---------|---------|
| `az-video` | Analyze a drone/helicopter video (main pipeline) |
| `az-image` | Generate enhancement variants from images |
| `az-llm` | Re-run LLM analysis on existing scene images |
| `az-previews` | Build hq/lq preview videos |
| `az-llm-preview` | Build LLM findings preview videos |
| `az-colors` | Color anomaly detection |
| `az-geometry` | Geometry anomaly detection |
| `az-info` | Report video file info |
| `az-batch` | Batch-run analysis on all videos in `source_video/` |
| `az-report` | Summarize findings from report.json files |
| `az-gdown` | Download a file/folder from Google Drive, or a video from YouTube |

Besides the package this installs all dependencies:

- `opencv-python` — frame I/O, image processing, CLAHE, phase correlation
- `numpy` — array operations
- `Pillow` — image I/O
- `piexif` — EXIF metadata embedding
- `gdown` — Google Drive folder downloads

### 2. FFmpeg

Required for video probing (`ffprobe`).

- **Windows:** Download from https://ffmpeg.org/download.html, add to PATH
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Verify:
```bash
ffprobe -version
```

### 3. Ollama

Ollama runs large language models locally, including vision-capable models 
used by this project for SAR image analysis.

#### Install Ollama

- **macOS:** Download installer from https://ollama.com
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows:** Download from https://ollama.com (native build)
- **Docker:** `docker pull ollama/ollama && docker run -d -v ollama:/root/.ollama -p 11434:11434 ollama/ollama`

Verify installation:
```bash
ollama --version
```

#### How Ollama works

Ollama is a lightweight runtime that downloads and serves LLMs via a local 
REST API at `http://localhost:11434`. Key concepts:

- **Pull a model:** `ollama pull <model>` — downloads to local storage
- **Run a model:** `ollama run <model>` — starts interactive chat (auto-pulls if missing)
- **List models:** `ollama list` — shows installed models
- **API server:** `ollama serve` — starts the API server (usually auto-starts on install)
- **Vision models:** accept image input via the `/api/chat` endpoint with base64-encoded 
  images in the message content

#### Cloud models

This project uses Ollama cloud models (e.g. `gemma4:31b-cloud`, `qwen3.5:397b-cloud`). 
These are hosted models accessed through the Ollama API with a cloud provider. 
No local GPU required — the model runs on remote infrastructure.

#### Verify model availability

```bash
ollama list
```

Ensure the fast and deep models are available. If not, pull them:
```bash
ollama pull gemma4:31b-cloud
ollama pull qwen3.5:397b-cloud
```

For two-stage mode, also pull the reasoning model:
```bash
ollama pull glm-5.1:cloud
```

### 4. Google Drive (optional)

For downloading footage from shared Google Drive folders:
```bash
pip install gdown
```

## Usage

### Analyze all videos
```bash
az-batch
```
No arguments. Processes all videos in `source_video/`, creates output in `analysis_results/`.
Skips videos with existing `report.json`. Auto-detects helicopter folders (HELI in name).

Batch runner uses overridden defaults: `--scene-sim 0.65` (more aggressive grouping),
`--llm-scenes-cap 50`, `--llm-deep-max-scenes 20`. Always passes `--llm-run` and
`--build-preview` to generate full results per video.

### Analyze single video
```bash
az-video <video> [options]
```
| Parameter | Default | Description |
|-----------|--------|-------------|
| `video` | (required) | Path to drone/heli video file |
| `-o, --output` | `<video_basename>.AZ/` | Output directory (defaults to the video filename + `.AZ`, e.g. `video1.mp4.AZ/`). The `.AZ` suffix avoids colliding with the source video file when running from the video's own directory. |
| `--stride` | `dynamic` | Frame sampling interval. Integer (every N frames) or `dynamic` (default, motion-adaptive). Dynamic reads every frame and adapts: stride 1 (every frame) during fast motion/zoom, up to 20 during slow pans. Zoom spike (>60px shift) forces immediate sample. Uses phase correlation to measure frame-to-frame shift. |
| `--quality` | `0.5` | Quality threshold (0-1). Frames scoring below this are filtered out. Score combines Laplacian variance (sharpness), phase correlation (camera motion), exposure, and contrast. Lower = more frames pass (less selective); higher = fewer but sharper frames. Helicopter mode auto-lowers to 0.4. Scenes below 0.4 quality are saved as images but not sent to LLM (budget protection). |
| `--scene-sim` | `0.82` | Scene similarity threshold (0-1). Frames are compared using normalized cross-correlation of 16×16 grayscale signatures. If a frame's similarity to the current scene's key frame exceeds this threshold, it's grouped into that scene. Lower (e.g. 0.65) = more distinct scenes (frames must be more different to start a new scene). Higher (e.g. 0.90) = fewer scenes (only very different frames start new scenes). Helicopter mode defaults to 0.55 (aggressive grouping due to fast camera movement). Timeline-local: only matches within 300 frames to prevent similar terrain at different times from merging. |
| `--llm-fast-model` | `gemma4:31b-cloud` | Fast LLM model for initial pass |
| `--llm-deep-model` | `qwen3.5:397b-cloud` | Deep LLM model for confirmation |
| `--llm-deep-max-scenes` | `20` | Max scenes to send to deep LLM (positives, or positives+chancepeek negatives) |
| `--llm-scenes-cap` | `50` | Max scenes to send to fast LLM (spread across timeline). Integer or `all`. |
| `--helicopter` | off | Helicopter mode (relaxed thresholds, auto-loads `sar-heli` context preset) |
| `--llm-pipeline` | `fast` | LLM pipeline mode: `fast` (fastest — v3 frame variant is assessed, stop-on-find, deep on positives), `chancepeek` (longer — deep also runs on fast negatives, last hope to catch false negatives), `max` (the longest, deepest assessment of all frame variants, no stop-on-find, deep on all variants for positives). |
| `--from` | None | Start processing from this time (seconds). Default: beginning |
| `--to` | None | Stop processing at this time (seconds). Default: end |
| `--llm-run` | off | Run LLM analysis (off by default — generates scenes/images only). Use `llm_analysis.py` to run LLM separately on existing results. |
| `--build-preview` | off | Build preview videos (hq/lq/color anomalies). Off by default to speed up batch processing. |
| `--dedup-thresh` | `0.90` | Scene deduplication threshold (PASS 2.5). Merges near-duplicate scenes across time gaps using 32×32 grayscale + color histogram similarity. Set to `0` to disable. Catches camera returning to same viewpoint minutes apart. |
| `--color-anomalies` | off | Enable scene-relative color anomaly detection. Finds small colored regions whose color is statistically rare for the scene (orange, blue, red, green, etc.). Uses LAB colorfulness + histogram rarity. Filters noise via min area (50px), min rarity (2.5), and ignore mask for text markers. Saves annotated images to `scenes/anomalies/` and builds `preview_color_anomalies.mp4` (orig + anomaly, 1s each). |
| `--recording-time` | None | Override recording time (ISO format, e.g. '2026-08-15 12:08:14'). Use when camera clock is wrong. |
| `--context-file` | None | Path to JSON mission context config. Overrides `--context-preset` and `--helicopter`. See `contexts/` for examples. |
| `--context-preset` | None | Named preset: `sar`, `sar-heli`. Overrides `--helicopter`. If neither set, `--helicopter` loads `sar-heli`. |
| `--llm-no-two-stage` | off | Disable two-stage LLM for the **deep** pass (on by default). The **fast** pass is always single-stage. Two-stage (deep only): vision model describes scene (no judgment), reasoning model concludes from description + context. Uses `--llm-deep-model` as vision, `--llm-reasoning-model` as reasoning. |
| `--llm-reasoning-model` | `glm-5.1:cloud` | Reasoning model for two-stage mode (text-only, no image). |
| `--llm-parallel` | `0` | Number of parallel LLM workers (0 = sequential). Cloud models can handle 4+ concurrent requests (~5x speedup). Local models should stay at 0 (VRAM-bound). See `LLM.PARALLELISM.md` for benchmark details. |
| `--run-standard` | off | Preset: full analysis with LLM. Sets `--color-anomalies --build-preview --llm-run --llm-parallel 4 --llm-pipeline chancepeek`. Individual flags override the preset. |
| `--run-light` | off | Preset: scenes + images + color anomalies + previews, no LLM. Sets `--color-anomalies --build-preview`. Individual flags override the preset. |

### Run color anomaly detection on existing results
```bash
az-colors <output_dir> [--debug] [--skip-update-report]
```
| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | (required) | Video output dir with report.json and scenes/ (like az-previews) |
| `--debug` | off | Save debug heatmaps (anomaly mask, colorfulness, rarity) |
| `--skip-update-report` | off | Do not write `dynamics.color_findings` back into report.json |

Scans `scenes/hq/` and `scenes/lq/` for `_orig.jpg` files, writes annotated
`_color_anomalies.jpg` images to `scenes/anomalies/`, and (by default) updates
`report.json` with `dynamics.color_findings` so `az-previews` can build
`preview_color_anomalies.mp4`. Uses the shared `build_dynamics()` helper so the
report structure matches `az-video --color-anomalies` exactly.

### Run geometry anomaly detection on existing results
```bash
az-geometry <output_dir> [--debug]
```
| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | (required) | Video output dir with report.json and scenes/ (like az-previews) |
| `--debug` | off | Save debug heatmaps (edges, contours) |

Detects man-made geometric shapes (lines, rectangles, circles, regular polygons)
that are statistically unlikely in natural terrain. Uses contour analysis, Hough
transforms, and gradient direction verification for circles. Scans `scenes/hq/`
and `scenes/lq/` for `_orig.jpg` files, writes annotated
`_geometry_anomalies.jpg` images to `scenes/anomalies/`.

### Build preview videos
```bash
az-previews <output_dir>
```
| Parameter | Description |
|-----------|-------------|
| `output_dir` | (required) Video output dir with report.json and scenes/ |

### Generate enhancement variants from a single image
```bash
az-image <image> [image2 ...] [options]
```
| Parameter | Default | Description |
|-----------|--------|-------------|
| `images` | (required) | One or more source image paths |
| `--llm-run` | off | Run LLM analysis (off by default — generates variants only unless specified) |
| `--llm-no-deep` | off | Skip deep LLM analysis (deep is auto-triggered by default when fast finds positives) |
| `--llm-force-deep` | off | Run deep LLM on ALL variants (not just fast-LLM positives) |
| `--helicopter` | off | Use helicopter-mode prompt (loads `sar-heli` preset) |
| `--context-file` | None | Path to JSON mission context config. Overrides `--context-preset` and `--helicopter`. |
| `--context-preset` | None | Named preset: `sar`, `sar-heli`. Overrides `--helicopter`. |
| `--llm-fast-model` | `gemma4:31b-cloud` | Fast LLM model |
| `--llm-deep-model` | `qwen3.5:397b-cloud` | Deep LLM model |
| `--llm-timeout` | `120` | LLM timeout in seconds |

Output structure:
```
scene_32_f00810/
  scene_32_f00810_grid_orig.jpg    # original + grid
  scene_32_f00810_grid_v1.jpg      # high contrast
  scene_32_f00810_grid_v2.jpg      # gentle CLAHE
  scene_32_f00810_grid_v3.jpg      # aggressive shadow recovery
  scene_32_f00810_grid_v4.jpg      # highlight recovery
  llm_analysis.txt                 # full LLM analysis report
```

When any fast variant finds positives, deep LLM is auto-triggered on those variants.
An overall summary is generated from all deep findings combined, starting with
"Found it." or "No Detection."

### Build LLM findings previews for existing results
```bash
az-llm-preview <root>
```
| Parameter | Description |
|-----------|-------------|
| `root` | (required) Root directory to scan for analysis results |

### Re-run LLM on existing scenes (no image regeneration)
```bash
az-llm <output_dir> [options]
```
| Parameter | Default | Description |
|-----------|--------|-------------|
| `output_dir` | (required) | Video output dir with report.json and scenes/ |
| `--helicopter` | off | Helicopter mode |
| `--llm-fast-model` | `gemma4:31b-cloud` | Fast LLM model |
| `--llm-deep-model` | `qwen3.5:397b-cloud` | Deep LLM model |
| `--llm-deep-max-scenes` | `20` | Max scenes to send to deep LLM |
| `--llm-fast-max-scenes` | `100` | Max scenes to analyze |
| `--llm-force-deep` | off | Run deep LLM on ALL scenes (not just fast-LLM positives) |
| `--llm-pipeline` | `fast` | LLM pipeline mode: `fast` (fastest), `chancepeek` (longer, last hope for false negatives), `max` (longest, deepest assessment of all frame variants). |
| `--context-file` | None | Path to JSON mission context config. Overrides preset/helicopter/stored context. |
| `--context-preset` | None | Named preset: `sar`, `sar-heli`. Overrides `--helicopter` and stored context. Falls back to context stored in `report.json` if set. |
| `--llm-no-two-stage` | off | Disable two-stage LLM for the **deep** pass (on by default). The **fast** pass is always single-stage. |
| `--llm-reasoning-model` | `glm-5.1:cloud` | Reasoning model for two-stage deep pass (text-only). |
| `--llm-parallel` | `0` | Number of parallel LLM workers (0 = sequential). 4 recommended for cloud models. |


### Report video info
```bash
az-info <video_path> [video_path2 ...]
```

### Download from Google Drive
```bash
# Download a folder
az-gdown dir "https://drive.google.com/drive/folders/FOLDER_ID" -o source_video/DATE

# Download a single file
az-gdown file "https://drive.google.com/file/d/FILE_ID/view" -o source_video/
```

### Download a YouTube video
Requires `yt-dlp` (`pip install yt-dlp`).
```bash
az-gdown youtube "https://www.youtube.com/watch?v=VIDEO_ID" -o output/
```

### Model comparison
| Model | Speed | Vision | Status |
|-------|-------|--------|--------|
| gemma4:31b-cloud | 6.5s | ✅ | Fast pass — all scenes (always single-stage) |
| qwen3.5:397b-cloud | 54.6s | ✅ | Deep pass — positives only / two-stage vision model |
| glm-5.1:cloud | ~3s | ❌ | Two-stage deep-pass reasoning model (text-only) |
| minimax-m3:cloud | 14.7s | ✅ | Available as backup |

## Output Structure

```
analysis_results/
  <date>/
    <video.MP4>/
      report.json              # Full analysis report with LLM + dynamics data
      contact_sheet.jpg        # Thumbnail grid of all scenes
       preview_hq.mp4           # HQ scenes preview (0.5s/image, 4 variants)
      preview_lq.mp4           # LQ scenes preview
      preview_llm_findings.mp4 # LLM findings preview (2s/image, text overlay)
      preview_color_anomalies.mp4 # Color anomalies preview (if --color-anomalies)
      scenes/
        hq/                    # High-quality scenes (sent to LLM)
          scene_00_f00000_orig.jpg
          scene_00_f00000_grid_v1.jpg
          scene_00_f00000_grid_v2.jpg
          scene_00_f00000_grid_v3.jpg
          scene_00_f00000_grid_v4.jpg
          ...
        lq/                    # Lower-quality scenes (not analyzed)
          ...
        anomalies/             # Color anomaly annotated images
          scene_00_f00000_color_anomalies.jpg
          ...
        llm_findings/          # Scenes with LLM findings
          ...
```
## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Copyright © 2026 Andrei Suvorov

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

Full license text: [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

**Additional permission — NonCommercial clarification:** Notwithstanding the NonCommercial clause, this work may be used by non-profit organizations, volunteer search-and-rescue groups, and government emergency services, including where such organizations accept donations or sponsorship to cover operational costs (e.g., computing infrastructure, GPU hosting, software development). This permission does not extend to commercial entities selling the software or offering it as a paid service.

### Disclaimer of Warranties

THIS WORK IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE WORK OR THE USE OR OTHER DEALINGS IN THE WORK.
