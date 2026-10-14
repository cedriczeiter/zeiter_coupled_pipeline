# Who Breathes the Detour? A Coupled Traffic–Emission–Dispersion–Exposure Pipeline

Code and data of the paper *Who Breathes the Detour? A Coupled Traffic–Emission–Dispersion–Exposure Pipeline*.
The pipeline couples microscopic traffic simulation (SUMO), per-vehicle emissions (HBEFA4), a three-dimensional dispersion solver and a resident-level air quality index. It is applied to the residential areas of six villages (Erstfeld, Göschenen, Gurtnellen, Schattdorf, Silenen, Wassen) along the Gotthard transit corridor in Canton Uri, Switzerland.

**Authors:** Cedric-Yséry Zeiter (ETH Zürich, Department of Mathematics), Kevin Riehl and Anastasios Kouvelas (ETH Zürich, Institute for Transport Planning and Systems, Traffic Engineering Group)

## Where to find what

| Folder | Purpose |
|---|---|
| [`source_code/`](source_code/) | The full pipeline. `route_matching/` turns detector data into SUMO traffic, `simulation/` runs traffic, emissions, dispersion and AQI post-processing (entry point `simulation/src/run_sim.py`), with tests in `simulation/method_testing/`. See [`source_code/README.md`](source_code/README.md). |
| [`reproducibility/`](reproducibility/) | Reproduces every figure of the paper from a small extract of the simulation output. Open `Reproduce_Paper_Figures.ipynb`. See [`reproducibility/README.md`](reproducibility/README.md). |
| [`toy_example/`](toy_example/) | A small, fast, runnable version of the pipeline on one village with a synthetic population, to try the code without the restricted input data. See [`toy_example/README.md`](toy_example/README.md). |

## Not in the repository

- The raw input data (ASTRA traffic counts, map data, the synthetic resident population), which is restricted or too large. The toy example replaces the population with a synthetic one, and `reproducibility/data/` holds everything needed for the figures.
