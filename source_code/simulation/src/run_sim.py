"""
Run one traffic + dispersion scenario for one village (run from simulation/src/).

    python run_sim.py --village <v> --date 07_20 --random-traffic 1 --weekday 0 \
                      --rerouting-percentage 0.05 [--wind 315:10]

All options: python run_sim.py --help. Settings that are not scenario flags are in config.py.
"""
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import time
from dataclasses import dataclass

import traci

import config as _cfg
from curated_manifest import sync_run_to_reproducibility, sync_wind_run_to_reproducibility
from dispersion_runner import (
    pollutants_from_config, init_fields, advance_fields, EmissionRecorder,
)
from emission_models import calculate_optimal_dt, get_emissions_batched, Z_THICKNESSES
from helper import get_cell_size, printv, parse_args
from houses import get_house_polygons
from import_people_data import get_people_data
from plotting import save_all_data
from post_processing import post_processing_wrapper
from sumo_commands import (
    startSumo, stopSumo, set_seeds,
    count_non_workers_non_students_with_car,
    count_workers_with_cars_adjusted,
    update_edge_speeds,
    reroute_vehicles_to_avoid_traffic,
    add_time_dependent_traffic,
    SPAWN_STATS,
)
from wind import WindSchedule, parse_wind_spec

OUTPUT_PATH_TEMP = "../output/temp"


@dataclass
class Scenario:
    random_traffic: bool
    weekday: bool
    rerouting_percentage: float
    wind: WindSchedule
    sim_time: float
    seed: int


@dataclass
class OutputFolders:
    root: str
    data: str
    plot: str
    avg: str
    final_results: str


def build_scenario(args):
    """Scenario flags from the CLI; the village/date overrides are applied to config."""
    _cfg.select_village(args.village, args.date)
    return Scenario(
        random_traffic=bool(args.random_traffic),
        weekday=bool(args.weekday),
        rerouting_percentage=args.rerouting_percentage,
        wind=parse_wind_spec(args.wind if args.wind is not None else _cfg.WIND_SPEC),
        sim_time=min(_cfg.TIME_PER_SCREENSHOT * 24, args.stop_hour * 3600),
        seed=_cfg.SEED if args.seed is None else args.seed,
    )


def print_settings(scenario):
    for label, value in [
        ("Current village", _cfg.VILLAGE_NAME), ("Rerouting period", _cfg.REROUTING_PERIOD),
        ("Time per screenshot", _cfg.TIME_PER_SCREENSHOT),
        ("Rerouting percentage", scenario.rerouting_percentage),
        ("Random traffic", scenario.random_traffic), ("Weekday", scenario.weekday),
        ("Force recalculation", _cfg.FORCE_RECALCULATE), ("Network file", _cfg.NETWORK_FILE),
        ("Seed", scenario.seed),
        ("Wind", scenario.wind.hourly if scenario.wind.active else "off"),
    ]:
        printv(f"{label + ':':<23}{value}", _cfg.VERBOSE, 'yellow')


def find_synpop_file():
    """SynPop CSV for both layouts (one or two levels above src/)."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    name = "synpop_clean_enriched.csv"
    candidates = [os.path.dirname(src_dir), os.path.dirname(os.path.dirname(src_dir))]
    return next((os.path.join(d, "data_cleaned", name) for d in candidates
                 if os.path.exists(os.path.join(d, "data_cleaned", name))),
                os.path.join(candidates[0], "data_cleaned", name))


def make_output_folders(scenario):
    stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wind_tag = ("_w" + "_".join(f"{d:g}-{v:g}" for d, v in sorted(set(scenario.wind.hourly.values())))
                if scenario.wind.active else "")
    prefix = "output_wind" if scenario.wind.active else "output"   # keeps wind runs out of the scenario-run globs
    root = (f"../{prefix}_{_cfg.VILLAGE_NAME}_{_cfg.DATE}_rt_{scenario.random_traffic}"
            f"_rp_{scenario.rerouting_percentage}_wd_{scenario.weekday}{wind_tag}_{stamp}/")
    folders = OutputFolders(root, f"{root}/data/", f"{root}/data/plot",
                            f"{root}/data/avg/", f"{root}/data/final_results/")
    for d in (folders.root, folders.data, folders.plot, folders.avg,
              folders.final_results, OUTPUT_PATH_TEMP):
        os.makedirs(d, exist_ok=True)
    return folders


def print_grid_info(cell_len_x_m, cell_len_y_m, scenario, dt):
    if not _cfg.VERBOSE:
        return
    print(f"Grid dimensions : {_cfg.GRID_DIM_X} x {_cfg.GRID_DIM_Y} x {_cfg.GRID_DIM_Z} (z-layers)")
    print(f"Cell size [m]   : x={cell_len_x_m:.2f}  y={cell_len_y_m:.2f}")
    print(f"Z-layer thicknesses [m]: {list(Z_THICKNESSES)}")
    print(f"Simulation      : {scenario.sim_time} s, screenshots every {_cfg.TIME_PER_SCREENSHOT} s")
    print(f"Timestep dt     : {dt} s (implicit, unconditionally stable)")


def load_population():
    """(people_coordinates_on_grid, working, with_car, inactive) for the village."""
    people = get_people_data(file_data=find_synpop_file(), file_sumo=_cfg.NETWORK_FILE,
                             recalculate=_cfg.FORCE_RECALCULATE, verbose=_cfg.VERBOSE)
    working, with_car = count_workers_with_cars_adjusted(people)
    inactive = count_non_workers_non_students_with_car(people)
    printv(f"People neither working nor studying:            {inactive}", True, "yellow")
    printv(f"Total people with cars:                         {with_car}", True, "yellow")
    printv(f"Workers with cars (FTE-weighted):               {working}", True, "yellow")
    return people, working, with_car, inactive


def load_house_polygons(people):
    return get_house_polygons(list_coords=list(set(people.keys())), netfile=_cfg.NETWORK_FILE,
                              recalculate=_cfg.FORCE_RECALCULATE, verbose=_cfg.VERBOSE,
                              output_dir=OUTPUT_PATH_TEMP)


def update_traffic(scenario, curr_time, dt, population, state):
    """Add random resident traffic for this step (if enabled) and reroute periodically."""
    working, with_car, inactive = population
    if scenario.random_traffic:
        add_time_dependent_traffic(scenario.weekday, curr_time, dt, with_car, working, inactive)
    vehicle_ids = list(traci.vehicle.getIDList())
    if state["next_rerouting"] <= curr_time:
        state["next_rerouting"] += _cfg.REROUTING_PERIOD
        if abs(scenario.rerouting_percentage) > 1e-6:
            reroute_vehicles_to_avoid_traffic(scenario.rerouting_percentage, vehicle_ids)
    return vehicle_ids


def save_snapshot(fields, folder, curr_time, cell_len_x_m, cell_len_y_m):
    save_all_data(
        co_vector_global=fields["co"], nox_vector_global=fields["nox"],
        pmx_vector_global=fields["pmx"],
        folder_path=folder, identifier=f"{curr_time}",
        cell_len_x_m=cell_len_x_m, cell_len_y_m=cell_len_y_m,
    )


def run_simulation_loop(scenario, population, folders, cell_len_x_m, cell_len_y_m, dt, recorder):
    """Advance SUMO and the dispersion fields to scenario.sim_time; returns the final fields."""
    pollutants = pollutants_from_config(_cfg)
    fields = init_fields(pollutants, _cfg.GRID_DIM_X, _cfg.GRID_DIM_Y, _cfg.GRID_DIM_Z,
                         cell_len_x_m, cell_len_y_m)
    state = {"next_rerouting": _cfg.REROUTING_PERIOD}
    next_snapshot = _cfg.TIME_PER_SCREENSHOT

    while traci.simulation.getTime() <= scenario.sim_time:
        curr_time = traci.simulation.getTime()
        traci.simulationStep()
        update_edge_speeds(curr_time)
        vehicle_ids = update_traffic(scenario, curr_time, dt, population, state)

        x_vec, y_vec, co_vec, nox_vec, pmx_vec = get_emissions_batched(vehicle_ids, dt)
        emissions = {"co": co_vec, "nox": nox_vec, "pmx": pmx_vec}
        if recorder is not None:      # recording mode: SUMO only, dispersion is replayed later
            recorder.add(curr_time, x_vec, y_vec, emissions)
            continue

        fields = advance_fields(fields, pollutants, x_vec, y_vec, emissions,
                                cell_len_x_m, cell_len_y_m, dt, wind=scenario.wind.at(curr_time))

        if next_snapshot <= curr_time:
            next_snapshot += _cfg.TIME_PER_SCREENSHOT
            print(f"{curr_time:.0f} / {scenario.sim_time:.0f} s")
            save_snapshot(fields, folders.data, curr_time, cell_len_x_m, cell_len_y_m)
    return fields


def finish(args, scenario, folders, recorder, wall_start):
    stopSumo()
    if scenario.random_traffic:
        print(f"Local vehicles: {SPAWN_STATS['spawned']} spawned, {SPAWN_STATS['failed']} failed "
              f"of {SPAWN_STATS['requested']} requested")
    wall = time.time() - wall_start
    print(f"Wall time (simulation, before post-processing): {int(wall // 3600)}h {int(wall % 3600 // 60):02d}m {wall % 60:04.1f}s")
    print("Closed SUMO")
    if recorder is not None:
        recorder.save(args.record_emissions)
        print(f"Recorded emissions to {args.record_emissions}")
        return
    post_processing_wrapper(
        data_folder=folders.data, plot_folder=folders.plot,
        avg_data_folder=folders.avg, final_results_folder=folders.final_results,
        verbose=_cfg.VERBOSE,
    )
    if scenario.wind.active:      # wind-sweep runs have their own curated folders
        winds = set(scenario.wind.hourly.values())
        if len(winds) == 1:       # the plotted sweep uses one constant wind per run
            direction, speed = next(iter(winds))
            sync_wind_run_to_reproducibility(
                village=_cfg.VILLAGE_NAME, date=_cfg.DATE, rt=scenario.random_traffic,
                rp=scenario.rerouting_percentage, wd=scenario.weekday,
                direction=direction, speed=speed, output_path=folders.root, verbose=_cfg.VERBOSE)
        return
    sync_run_to_reproducibility(
        village=_cfg.VILLAGE_NAME, date=_cfg.DATE, rt=scenario.random_traffic,
        rp=scenario.rerouting_percentage, wd=scenario.weekday,
        output_path=folders.root, house_poly_dir=OUTPUT_PATH_TEMP, verbose=_cfg.VERBOSE,
    )


def main():
    wall_start = time.time()
    args = parse_args()
    scenario = build_scenario(args)
    print_settings(scenario)
    dt = calculate_optimal_dt()
    folders = make_output_folders(scenario)
    printv("Simulation started", _cfg.VERBOSE, 'magenta', True)
    set_seeds(scenario.seed)
    with open(os.path.join(folders.root, "seed.txt"), "w") as f:
        f.write(f"seed={scenario.seed}\n")   # used for SUMO --seed, Python random and NumPy

    cell_len_x_m, cell_len_y_m = get_cell_size(
        grid_left=_cfg.GRID_LEFT, grid_right=_cfg.GRID_RIGHT,
        grid_bottom=_cfg.GRID_BOTTOM, grid_top=_cfg.GRID_TOP,
        size_x=_cfg.GRID_DIM_X, size_y=_cfg.GRID_DIM_Y,
        filename=_cfg.NETWORK_FILE, verbose=True)
    print_grid_info(cell_len_x_m, cell_len_y_m, scenario, dt)

    people, working, with_car, inactive = load_population()
    load_house_polygons(people)   # writes house_polygons_*.csv to the temp folder (used by run_plotting.py)

    startSumo(200, _cfg.GRID_LEFT, _cfg.GRID_TOP, visual_interface=_cfg.SHOW_INTERFACE,
              verbose=_cfg.VERBOSE, dt=dt, seed=scenario.seed)
    recorder = EmissionRecorder() if args.record_emissions else None
    run_simulation_loop(scenario, (working, with_car, inactive), folders,
                        cell_len_x_m, cell_len_y_m, dt, recorder)
    print("Simulation finished")
    finish(args, scenario, folders, recorder, wall_start)


if __name__ == "__main__":
    main()
