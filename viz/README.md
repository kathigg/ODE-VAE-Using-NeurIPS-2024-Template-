# Runs Visualization (Local Dashboard)

This is a lightweight, dependency-free dashboard to browse `neuroscience/runs/*` artifacts (metrics, images, and learning curves parsed from `run.log`).

It’s meant to feel more like a “playground” for your experiments: you can scrub epochs, play them back like a video, inspect configs/logs, and export a short `.webm` recording.

## Start

From the repo root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/viz/
```

## What you get

- Run list + filter (reads `../neuroscience/runs/index.csv`)
- Key run/config details
- Model architecture (layer-level “playground” view) + objective summary (from `config.txt`, plus dimensions parsed from `run.log`)
- Learning curve playback (epoch slider + Play/Pause)
- Approx loss composition (uses `lambda_*` from `config.txt` + `beta` from `run.log`)
- File viewer (`run.json`, `config.txt`, `config_original.txt`, `run.log`, `git_diff.patch`)
- Artifacts (`preview.png`, `latent_manifold_mds.png` when present)
- Video export: click **Record Video** to download a `.webm` (and **Snapshot PNG** for a still frame)

## Notes

- The dashboard reads files from `../neuroscience/runs/` (relative to `viz/`), so it won’t work on platforms that don’t include the git submodule (e.g., Overleaf) unless you copy the run artifacts into the Overleaf project.
- If a run is missing an image (e.g., `preview.png`), that panel will just show “(missing)”.
