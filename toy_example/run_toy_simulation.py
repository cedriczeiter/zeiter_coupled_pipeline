"""
A minimal, readable stand-in for simulation/src/run_sim.py: it calls the
exact same pipeline functions (SUMO traffic -> per-vehicle emissions ->
3-D diffusion -> hourly concentration snapshots -> AQI post-processing),
just for a short simulated window on one small village, using a synthetic
population instead of the real (unpublishable) SynPop data.

What is deliberately left out compared to the full pipeline, for simplicity
(find the full code in source_code/simulation/src/):
  - randomised local traffic / dynamic rerouting
  - house-polygon plotting (get_house_polygons() geocodes building outlines
    over the internet; AQI itself does not need it, see below)

What the synthetic population is used for: after the AQI grids are computed,
each synthetic resident's coordinates are mapped to a grid cell and the AQI
there is read off. The per-pollutant values are summed without weighting, as
in the paper's per-resident index (so the number is a relative measure, not an
EPA AQI). Only pollutants with enough hourly history are included: NO2 needs
1 h, CO 8 h, PM2.5 24 h, so run 24 simulated hours for the full three-pollutant
index. The result is dominated by the background floor in this tiny run.

Usage:
    python run_toy_simulation.py [sim_hours]   # default: 24 (~6 min); 10 is enough for a quick look
"""
import os
import sys

SIM_HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
# Goeschenen is the quietest of the six villages: only a few vehicles per hour
# even at the busiest times (~170 in the first 10 h). Snapshots are hourly, matching run_sim.py, because
# the AQI step below needs hourly-spaced data for its rolling averages.
SCREENSHOT_EVERY = 3600

TOY_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_SRC = os.path.join(TOY_DIR, "..", "source_code", "simulation", "src")
sys.path.insert(0, REAL_SRC)

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from helper import get_cell_size
from emission_models import calculate_optimal_dt, get_emissions_batched
from dispersion_runner import pollutants_from_config, init_fields, advance_fields
from import_people_data import get_people_data
from sumo_commands import startSumo, stopSumo, update_edge_speeds
from plotting import save_all_data
import config as _cfg
import numpy as np

# ---------------------------------------------------------------------------
# Point the real pipeline's config at our toy village_data/ + fake population,
# instead of simulation/villages/ and data_cleaned/synpop_clean_enriched.csv.
# select_village() sets the grid (GRID_LEFT/RIGHT/TOP/BOTTOM, GRID_DIM_X/Y) of
# the village; the file paths are then replaced by the toy copies.
# ---------------------------------------------------------------------------
VILLAGE_DIR = os.path.join(TOY_DIR, "village_data", "Goeschenen")
_cfg.select_village("goeschenen", "20_11")
_cfg.NETWORK_FILE  = os.path.join(VILLAGE_DIR, "goeschenen_osm.net.xml")
_cfg.SUMO_FILE     = os.path.join(VILLAGE_DIR, "20_11_goeschenen.sumocfg")
_cfg.SPEED_FILE    = os.path.join(VILLAGE_DIR, "20_11_exit_speeds_per_second_goeschenen.csv")
_cfg.SHOW_INTERFACE = False
_cfg.VERBOSE = True

FAKE_POPULATION_FILE = os.path.join(TOY_DIR, "fake_population.csv")
if not os.path.exists(FAKE_POPULATION_FILE):
    raise FileNotFoundError(
        "fake_population.csv not found -- run generate_fake_population.py first."
    )

OUTPUT_DIR  = os.path.join(TOY_DIR, "toy_output")
DATA_FOLDER = os.path.join(OUTPUT_DIR, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

GRID_LEFT, GRID_RIGHT = _cfg.GRID_LEFT, _cfg.GRID_RIGHT
GRID_BOTTOM, GRID_TOP = _cfg.GRID_BOTTOM, _cfg.GRID_TOP
GRID_DIM_X, GRID_DIM_Y = _cfg.GRID_DIM_X, _cfg.GRID_DIM_Y

SIM_TIME = SIM_HOURS * 3600
dt = calculate_optimal_dt()

print(f"Toy simulation: Goeschenen, {SIM_HOURS:.1f}h, snapshots every {SCREENSHOT_EVERY}s")
print(f"Grid: {GRID_DIM_X} x {GRID_DIM_Y}, dt={dt}s")

cell_len_x_m, cell_len_y_m = get_cell_size(
    grid_left=GRID_LEFT, grid_right=GRID_RIGHT,
    grid_bottom=GRID_BOTTOM, grid_top=GRID_TOP,
    size_x=GRID_DIM_X, size_y=GRID_DIM_Y,
    filename=_cfg.NETWORK_FILE, verbose=True,
)

# Same per-species setup as run_sim.py: background, Dx/Dy and loss rate lambda
# come from config.py; Kz is the Monin-Obukhov profile (Kz=None).
pollutants = pollutants_from_config(_cfg)
fields = init_fields(pollutants, GRID_DIM_X, GRID_DIM_Y, _cfg.GRID_DIM_Z,
                     cell_len_x_m, cell_len_y_m)

# ---------------------------------------------------------------------------
# get_people_data() snaps residents to grid cells and caches the result as a
# .pkl in ../output/temp relative to the working directory, hence the chdir into
# _workdir (the cache ends up in toy_example/output/temp). The residents are used
# for the exposure step at the end; the random local traffic that run_sim.py
# derives from them is not part of this toy.
# ---------------------------------------------------------------------------

_workdir = os.path.join(TOY_DIR, "_workdir")
os.makedirs(_workdir, exist_ok=True)
os.chdir(_workdir)
people_coordinates_on_grid = get_people_data(
    file_data=FAKE_POPULATION_FILE, file_sumo=_cfg.NETWORK_FILE,
    recalculate=True, verbose=True,
)
print(f"Synthetic residents inside the village grid: "
      f"{sum(v['count'] for v in people_coordinates_on_grid.values())}")

# ---------------------------------------------------------------------------
# Main SUMO <-> emissions <-> dispersion loop
# ---------------------------------------------------------------------------
startSumo(200, GRID_LEFT, GRID_TOP, visual_interface=False, verbose=True, dt=dt)

next_screenshot_time = SCREENSHOT_EVERY
try:
    while traci.simulation.getTime() <= SIM_TIME:
        curr_time = traci.simulation.getTime()
        traci.simulationStep()
        update_edge_speeds(curr_time)

        vehicle_ids = list(traci.vehicle.getIDList())
        x_vec, y_vec, co_vec, nox_vec, pmx_vec = get_emissions_batched(vehicle_ids, dt)

        emissions = {"co": co_vec, "nox": nox_vec, "pmx": pmx_vec}
        fields = advance_fields(fields, pollutants, x_vec, y_vec, emissions,
                                cell_len_x_m, cell_len_y_m, dt)

        if next_screenshot_time <= curr_time:
            next_screenshot_time += SCREENSHOT_EVERY
            print(f"{curr_time:.0f} / {SIM_TIME:.0f} s  ({len(vehicle_ids)} vehicles)")
            save_all_data(
                co_vector_global=fields["co"], nox_vector_global=fields["nox"],
                pmx_vector_global=fields["pmx"],
                folder_path=DATA_FOLDER, identifier=f"{curr_time}",
                cell_len_x_m=cell_len_x_m, cell_len_y_m=cell_len_y_m,
            )
finally:
    stopSumo()

print(f"Snapshots written to {DATA_FOLDER}")

# ---------------------------------------------------------------------------
# AQI post-processing: the same two functions post_processing_wrapper() calls
# in run_sim.py, minus plot_all_heatmaps(), which only draws house-outline
# PNGs and needs house_polygons for that. The AQI computation itself
# (process_data_statistics + process_data_annoyance) is pure grid math and
# needs no house or population data at all.
#
# It does need enough hourly snapshots for its rolling-average windows: NO2
# needs 1 hour, CO needs 8, PM2.5 needs the full 24. A shorter SIM_HOURS will
# skip whichever pollutants don't have enough history yet.
# ---------------------------------------------------------------------------
from process_data import process_data_statistics
from process_annoyance import process_data_annoyance

AVG_FOLDER = os.path.join(OUTPUT_DIR, "data", "avg")
FINAL_RESULTS_FOLDER = os.path.join(OUTPUT_DIR, "data", "final_results")
os.makedirs(AVG_FOLDER, exist_ok=True)
os.makedirs(FINAL_RESULTS_FOLDER, exist_ok=True)

process_data_statistics(DATA_FOLDER, avg_data_folder=AVG_FOLDER, verbose=False)
process_data_annoyance(data_folder=DATA_FOLDER, final_results_folder=FINAL_RESULTS_FOLDER,
                       processed_data_folder=AVG_FOLDER, verbose=False)

available = {p: os.path.join(FINAL_RESULTS_FOLDER, f"{p}_max_aqi.csv")
             for p in ["co", "nox", "pmx"]}
available = {p: f for p, f in available.items() if os.path.exists(f)}
print(f"AQI computed for: {list(available.keys()) or 'none (need more simulated hours)'}")

# ---------------------------------------------------------------------------
# Resident exposure: map each synthetic resident's coordinates to a grid
# cell and read the AQI there, summed over whichever pollutants are
# available. This is what the synthetic population is actually for.
# ---------------------------------------------------------------------------
if available:
    import pandas as pd

    dx = (GRID_RIGHT - GRID_LEFT) / GRID_DIM_X
    dy = (GRID_TOP - GRID_BOTTOM) / GRID_DIM_Y
    grids = {p: pd.read_csv(f).values for p, f in available.items()}

    resident_aqi = []
    for x, y in people_coordinates_on_grid:
        xi = min(int((x - GRID_LEFT) / dx), GRID_DIM_X - 1)
        yi = min(int((y - GRID_BOTTOM) / dy), GRID_DIM_Y - 1)
        n = people_coordinates_on_grid[(x, y)]["count"]
        total_aqi = sum(g[xi, yi] for g in grids.values())
        resident_aqi.extend([total_aqi] * n)

    resident_aqi = np.array(resident_aqi)
    print(f"Residents matched to the grid: {len(resident_aqi)}")
    print(f"AQI per resident (sum of {list(available.keys())}): "
          f"mean={resident_aqi.mean():.1f}, min={resident_aqi.min():.1f}, max={resident_aqi.max():.1f}")

print("Done.")
