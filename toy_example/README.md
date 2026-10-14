# Toy example

A small, fast, plug-and-play version of the paper's pipeline: real SUMO
traffic simulation, real per-vehicle emission model, real 3-D diffusion
scheme (the same `dispersion_runner` helpers as `run_sim.py`) on one village (Goeschenen), with a **synthetic** resident
population standing in for the real (unpublishable) SynPop dataset.

## Quick start

```bash
cd toy_example
pip install -r requirements.txt   # use a fresh venv/conda env
python generate_fake_population.py 500 fake_population.csv
python run_toy_simulation.py 10.0   # simulated hours; ~2-3 min for 10 h, ~6 min for 24 h
```

Or open **`Toy_Example.ipynb`** and run it top to bottom it does the same
two steps and plots the resulting CO/NO2/PM2.5 excess over background.

## Layout

| Path | Contents |
|---|---|
| `village_data/Goeschenen/` | Real network, SUMO config, routes and exit-speed schedule for Goeschenen, 20 Nov (the smallest village/date pair, for a fast run) |
| `generate_fake_population.py` | Builds a synthetic resident population with the exact CSV schema the pipeline expects, from hand-picked category proportions no real data involved |
| `run_toy_simulation.py` | Runs the real pipeline functions (`get_people_data`, `get_emissions_batched`, `advance_fields`, `save_all_data`, `process_data_statistics`, `process_data_annoyance`) end to end, then reads the AQI at each synthetic resident. Without random traffic, rerouting or house-outline plots |
| `Toy_Example.ipynb` | Narrated walkthrough of both scripts, with plots |

## What to expect

Goeschenen is the quietest village: only about 170 vehicles are simulated in
the first 10 h, so concentrations rise only a fraction of a percent above
background (CO ~0.1 %). The notebook therefore plots the *excess over
background*; absolute maps would look uniform.

Output goes to `toy_output/data/` (hourly snapshots `data_<gas>_<layer>_<t>.csv`,
`avg/`, `final_results/`) and is not tracked by git. The final printout is the
mean summed AQI per resident. The AQI step needs 1 h of history for NO2, 8 h
for CO and 24 h for PM2.5, so a 10 h run covers CO and NO2 only; use
`python run_toy_simulation.py 24` for the full three-pollutant index.

The scripts import the real code from `../source_code/simulation/src/`, so
that folder must be present alongside `toy_example/`.

## Adapting this to another village

1. Copy that village's `*_osm.net.xml`, `<date>_<village>.sumocfg`,
   `<date>_routes_<village>.rou.xml`, `<date>_exit_speeds_per_second_<village>.csv`
   and `sumo_in_out_points.add.xml` into `village_data/<Village>/`
   (from `../source_code/simulation/villages/<Village>/`).
2. Update `NETFILE` in `generate_fake_population.py` and, at the top of `run_toy_simulation.py`,
   `VILLAGE_DIR`, the village name and date in `_cfg.select_village(...)` and the file names in the
   `_cfg.*` overrides (the grid of each village is defined in `VILLAGE_CONFIGS` in `config.py`).
