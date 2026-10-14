# Reproducibility of paper figures

Reproduces the figures of `paper.tex` from a small data extract.

## Usage

`pip install -r requirements.txt`, then run **`Reproduce_Paper_Figures.ipynb`** top to bottom. It regenerates all paper figures into `plotting/figures/`. The first run downloads the terrain tiles for the study-area figure (internet needed).

## Contents

- `plotting/` : one script per figure (they read `data/`; each can also be run on its own from `plotting/`), `_shared.py` with the common loaders and the heatmap primitive
- `data/` : extract of the simulation outputs, see below

## `data/`

An extract used only for reproducing the paper figures. A full simulation run writes far more CSVs; only the files the plotting scripts read are kept here, taken from the newest run of each configuration.

- `output_<village>_<date>_rt_<random traffic>_rp_<rerouting share>_wd_<weekday>_curated/data/` : one folder per simulation scenario (extract of one run):
  - `data_<gas>_1_<seconds>.csv` : concentration grid of one pollutant (`co`, `nox`, `pmx`) in the 0.3-1.2 m layer at one time of day (e.g. `64800.0` = 6 PM), in mg/m³. One cell per row/column entry, about 10 m × 10 m.
  - `final_results/*_max_aqi.csv` : per-cell maximum AQI over the day, per pollutant and `overall`; `*_max_aqih.csv` is the same with the AQIH health index. Same grid shape.
- `output_wind_schattdorf_..._w<direction>-<speed>_curated/` : same format, one wind-sweep run each (9 AM CO snapshot only).
- `converted_house_polygons_*.csv` : residential building footprints of one village (OpenStreetMap), one row per building: `lat`, `lon`, SUMO coordinates `sumo_x`/`sumo_y` and the footprint polygon in SUMO coordinates.
- `residents_<village>.pkl` : Python list with one `(grid_x, grid_y)` entry per resident: the grid cell of that resident's home. Contains no other information about people.
- `study_area_basemap/` : base map for the study-area figure: OpenStreetMap (© contributors, ODbL) and Natural Earth. The terrain tiles (Mapzen Terrain Tiles, Copernicus EU-DEM) are downloaded by the notebook and not part of the repository.

`data/` is filled by `source_code/simulation/src/curated_manifest.py`, which `run_sim.py` calls after each run.
