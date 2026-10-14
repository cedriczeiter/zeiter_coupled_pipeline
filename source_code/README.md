# Source code

Pipeline: **route matching → simulation → post-processing**. The six villages (Erstfeld, Göschenen, Gurtnellen, Schattdorf, Silenen, Wassen) are in Canton Uri.

## `route_matching/`
Reconstructs the through-traffic of one day from the detector data and prepares the SUMO input.
- `parse_routes.py`: matches entry and exit detections of each vehicle to trips
- `from_route_to_trip.py`: writes the trips as SUMO trip files
- `speed_regulation.py`: per-second speed of every exit lane, used as the speed limit of the exit lanes in the simulation
- `run_sumo_tool.sh [DATE] [SENSOR_DIR] [OUT_DIR]`: runs the three scripts and SUMO's `duarouter` for all villages (default: 20 Nov)
- `data/`: vehicle types and the detector data of the three dates (`czeiter_loop_output*`); `output_17_05/`, `output_20_07/`: results of earlier matching runs

## `simulation/`
Traffic simulation, emission and dispersion, and post-processing. Run everything from `simulation/src/`.
- `src/`
  - `run_sim.py`: entry point, one scenario per call (`python run_sim.py --help`)
  - `config.py`: settings (grid of each village, diffusion coefficients, background concentrations, seed)
  - `emission_models.py`, `atmosphere.py`, `wind.py`, `dispersion_runner.py`: per-vehicle emissions and the 3-D dispersion solver
  - `sumo_commands.py`: SUMO interface, local random traffic, rerouting
  - `import_people_data.py`, `houses.py`: resident population and house outlines (OpenStreetMap)
  - `process_data.py`, `process_annoyance.py`, `post_processing.py`: time averages and AQI per cell
  - `plotting.py`, `run_plotting.py`: heatmaps; `run_plotting.py <output folder>` makes the paper-style plots of a finished run
  - `curated_manifest.py`: copies the files the paper figures need from a finished run into `reproducibility/data/`
- `method_testing/`: tests of the dispersion solver, the vertical diffusivity and wind, and the AQI code (`python -m pytest`)
- `villages/`: per-village SUMO network, routes and exit speeds for each date
- `requirements.txt`: Python dependencies (SUMO itself is needed too)

## Data
The input data (raw ASTRA traffic, map and population data, and the cleaned population data) is not part of the repository. `run_sim.py` expects the cleaned population as `source_code/data_cleaned/synpop_clean_enriched.csv`. Simulation output is written to `simulation/output_*/` and is not tracked.
