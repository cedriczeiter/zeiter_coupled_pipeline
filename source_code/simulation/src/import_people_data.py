import os
import pickle
from collections import defaultdict

from helper import convert_lat_lon_to_sumo_coordinates, printv
import config as _cfg

CACHE_FOLDER = "../output/temp/"

# Column positions in synpop_clean_enriched.csv (same layout as toy_example/generate_fake_population.py)
COL_AGE, COL_EMPLOYMENT, COL_INCOME, COL_EDU, COL_JOB, COL_CAR_TYPE, COL_PT = range(7)
COL_LAT, COL_LON, COL_HAS_CAR = 9, 10, 11


def get_people_data(file_data, file_sumo, recalculate=False, verbose=True):
    """
    Get the residents inside the current village grid from the SynPop CSV.
    The result is cached per grid in a pickle file and reused unless `recalculate` is set.

    Args:
        file_data (str): Path to the input file.
        file_sumo (str): Path to the SUMO network file.
        recalculate (bool): Whether to force recalculation of data
        verbose (bool): If more information should be printed.

    Returns:
        dict: Dictionary with keys (x_coord, y_coord)->sumo system, and values as dictionaries containing:
            - 'count': Number of people at this coordinate
            - 'people': List of dictionaries containing people's attributes
    """

    printv("Start function get_people_data",
           verbose=verbose, color="blue", decorate=True)

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    cache_filename = (f"{CACHE_FOLDER}cached_people_data_"
                      f"{_cfg.GRID_LEFT}_{_cfg.GRID_RIGHT}_{_cfg.GRID_BOTTOM}_{_cfg.GRID_TOP}.pkl")

    if os.path.exists(cache_filename) and not recalculate:
        printv(f"Loading cached people data from '{cache_filename}'...", verbose=verbose)
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)

    with open(file_data, 'r') as f:
        data = f.readlines()[1:]  # skip header

    n = len(data)
    people_data = defaultdict(lambda: {"count": 0, "people": []})

    printv("Start of reading people... can take a while", verbose=verbose)

    next_report = 0
    for counter, line in enumerate(data, start=1):
        if verbose and counter / n * 100 >= next_report:
            printv(f"Scanned people: {next_report} %", verbose=verbose)
            next_report += 10

        parts = line.strip().split(',')

        x_coord, y_coord = convert_lat_lon_to_sumo_coordinates(
            float(parts[COL_LAT]), float(parts[COL_LON]), file_sumo)

        if (_cfg.GRID_LEFT <= x_coord <= _cfg.GRID_RIGHT) and (_cfg.GRID_BOTTOM <= y_coord <= _cfg.GRID_TOP):
            person_dict = {
                "age": parts[COL_AGE],
                "level_of_employment": parts[COL_EMPLOYMENT],
                "household_income": parts[COL_INCOME],
                "position_in_edu": parts[COL_EDU],
                "position_in_bus": parts[COL_JOB],
                "cars_drivetype": parts[COL_CAR_TYPE],
                "public_transport": parts[COL_PT],
                "has_car_and_licence": parts[COL_HAS_CAR],
                "x_coord": x_coord,
                "y_coord": y_coord
            }

            coord_key = (x_coord, y_coord)
            people_data[coord_key]["count"] += 1
            people_data[coord_key]["people"].append(person_dict)

    printv("End of reading people", verbose=verbose)

    result = dict(people_data)

    with open(cache_filename, 'wb') as f:
        pickle.dump(result, f)
    printv(f"Saved processed people data to '{cache_filename}'", verbose=verbose)

    return result
