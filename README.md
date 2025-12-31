# GPX to Markdown

Convert a mountain bike GPX track into French Markdown artifacts (summary, turns, climbs, pauses, and POIs) enriched with OpenStreetMap via OSRM and Overpass.

## Features

- OSRM map matching for road/trail names
- POI enrichment from OSM (villages, peaks, churches, rivers, lakes, forests, windmills, power lines, rails, bridges, tunnels, roads, fields)
- Human-friendly summary mode (polyline simplification + turn clustering)
- Turn detection (angle > 80°)
- Climb detection (>= 4% grade, estimated duration > 3 minutes at 8 km/h)
- Pause detection (speed ~0 for > 5 minutes)
- Output in French Markdown with km and timestamps

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python gpx_to_markdown.py path/to/track.gpx --output summary.md
```

### Options

- `--osrm-url` (default: `https://router.project-osrm.org`)
- `--osrm-profile` (default: `driving`)
- `--overpass-url` (default: `https://overpass-api.de/api/interpreter`)
- `--turn-angle` (default: `80`)
- `--min-grade` (default: `4`)
- `--climb-min-minutes` (default: `3`)
- `--verbosity` (default: `human`, or `detailed`)
- `--simplify-tolerance` (default: `20`)
- `--turn-min-spacing` (default: `120`)
- `--turn-cluster-radius` (default: `200`)
- `--poi-radius` (default: `100`)
- `--osrm-max-points` (default: `1000`)
- `--output` (default: stdout)

## Output

Example Markdown bullets:

```
- [km 3.2 | 2024-06-12 09:14] Tourner à droite sur "Chemin des Crêtes" (≈92°)
- [km 4.1–6.0 | 2024-06-12 09:20–09:34] Montée 180 m sur 1.9 km (≈14 min) sur "GR7"
- [km 6.3 | 2024-06-12 09:37] Traverser le village "X"
- [km 7.0 | 2024-06-12 09:42] Passer près de l'église "Y"
- [km 9.5 | 2024-06-12 10:05–10:11] Pause détectée (≈6 min)
```

## Notes

- If OSRM matching fails, the script falls back to raw GPX geometry and emits a warning.
- Overpass responses are cached locally in `.gpx_to_markdown_cache.json`.
