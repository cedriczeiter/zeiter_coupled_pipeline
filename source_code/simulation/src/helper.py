import argparse
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sumolib import net


@lru_cache(maxsize=4)
def _read_net(filename):
    """Parse a SUMO network once; the conversions below are called per person / per polygon vertex."""
    return net.readNet(filename)


def convert_lat_lon_to_sumo_coordinates(lat, lon, filename, verbose=False):
    """
    Convert latitude and longitude to SUMO coordinates.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        filename (str): Path to the SUMO network file.
        verbose(bool): If more stuff should be printed

    Returns:
        tuple: SUMO coordinates (x, y).
    """
    assert (feasible_lat_long(lat, lon))
    printv(f"Start function convert_lat_lon_to_sumo_coordinates for {lat} and {lon}",
           verbose=verbose, color="blue")

    x, y = _read_net(filename).convertLonLat2XY(lon, lat)
    return x, y


def convert_sumo_coordinates_to_lat_lon(x, y, filename, verbose=False):
    """
    Convert SUMO coordinates to latitude and longitude.

    Args:
        x (float): SUMO X coordinate.
        y (float): SUMO Y coordinate.
        filename (str): Path to the SUMO network file.
        verbose(bool): If more stuff should be printed

    Returns:
        tuple: Latitude and longitude in degrees.
    """

    printv(f"Start function convert_sumo_coordinates_to_lat_lon for {x} and {y}",
           verbose=verbose, color="blue")

    lon, lat = _read_net(filename).convertXY2LonLat(x, y)

    assert (feasible_lat_long(lat, lon))

    return lat, lon


def parse_args():
    """
    Parses command line arguments for the traffic simulation
    """

    parser = argparse.ArgumentParser(description='Run traffic simulation')

    parser.add_argument('--random-traffic', type=int, default=0, choices=[0, 1],
                      help='Enable random traffic patterns (binary: 0=off, 1=on, default: 0)')
    parser.add_argument('--weekday', type=int, default=0, choices=[0, 1],
                      help='Day type (binary: 0=weekend, 1=weekday, default: 0)')
    parser.add_argument('--rerouting-percentage', type=float, default=0.0,
                      help='Percentage of vehicles that will reroute (default: 0.0)')
    parser.add_argument('--village', type=str, default=None,
                      choices=['erstfeld','goeschenen','gurtnellen',
                               'schattdorf','silenen','wassen'],
                      help='Village to simulate (overrides config.py). Required for batch runs.')
    parser.add_argument('--date', type=str, default=None,
                      choices=['07_20', '20_11', '05_17'],
                      help='Simulation date MM_DD (overrides config.py).')
    parser.add_argument('--wind', type=str, default=None,
                      help='Optional uniform wind, direction the wind blows FROM in degrees and speed '
                           'in m/s: "315:10" (all day) or per hour "8-17=315:10;18=100:3". '
                           'Default: off (calm).')
    parser.add_argument('--seed', type=int, default=None,
                      help='Seed for SUMO, Python random and NumPy (default: config.SEED = 42).')
    parser.add_argument('--record-emissions', type=str, default=None,
                      help='Write every step\'s vehicle positions/emissions to this .npz '
                           '(for fast dispersion replays); stops at --stop-hour.')
    parser.add_argument('--stop-hour', type=float, default=24.0,
                      help='Stop the simulation at this hour of the day (default: 24).')

    return parser.parse_args()


def feasible_lat_long(lat, long, verbose=False):
    """
    Check if the given latitude and longitude are within the bounds of Switzerland.

    Args:
        lat (float): Latitude in degrees.
        long (float): Longitude in degrees.
        verbose(bool): If more stuff should be printed

    Returns:
        bool: True if the coordinates are within Switzerland, False otherwise.
    """
    printv("Start function feasible_lat_long", verbose=verbose, color="blue")
    return (lat >= 40 and lat <= 50) and (long >= 5 and long <= 10)


def printv(message, verbose=True, color=None, decorate=False, info=True):
    """
    Print a message if verbose is True, optionally with colored output and decorative lines.

    Args:
        message (str): The message to print.
        verbose (bool): Whether to print the message.
        color (str or None): One of ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'] (case-insensitive).
        decorate (bool): If True, print a separator line before and after the message.
        info (bool): Whether "[info]" is put in front of the message (ignored if decorated)
    """

    if not verbose:
        return

    color_codes = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '36',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }

    if info and not decorate:
        message = f'[info] {message}'

    lines = "-" * len(message)

    def apply_color(text, color_code):
        return f"\033[{color_code}m{text}\033[0m"

    # Print the message with the specified color
    if color and color.lower() in color_codes:
        code = color_codes[color.lower()]
        if decorate:
            print(apply_color(lines, code))
        print(apply_color(message, code))
        if decorate:
            print(apply_color(lines, code))
    else:
        if decorate:
            print(lines)
        print(message)
        if decorate:
            print(lines)


def save_vec_to_csv(data_array, data_folder, filename):
    """
    Save statistics array to a CSV file

    Args:
        data_array: NumPy array of data
        data_folder: Path to the data folder
        filename: Name of the output file
    """
    df = pd.DataFrame(data_array)
    output_file = os.path.join(data_folder, filename)
    df.to_csv(output_file, index=False)


def get_cell_size(grid_left, grid_right, grid_bottom, grid_top, size_x, size_y, filename, verbose=False):
    """
    Calculate the size in real life [m] of each cell in the grid.

    Args:
        grid_left, grid_right, grid_bottom, grid_top (float): boundaries of the grid in sumo coords.
        size_x, size_y (unsigned int): dimensions of our grid
        filename (str): Path to the SUMO network file.
        verbose (bool): If more information should be printed.

    Returns:
        Tuple of cell width and height [m]
    """

    assert (size_x > 0 and size_y > 0)

    bottom_left_lat, bottom_left_lon = convert_sumo_coordinates_to_lat_lon(
        grid_left, grid_bottom, filename, verbose=verbose)
    top_right_lat, top_right_lon = convert_sumo_coordinates_to_lat_lon(
        grid_right, grid_top, filename, verbose=verbose)

    # Calculate the distances between the corners of the grid

    lat_diff = top_right_lat - bottom_left_lat
    lon_diff = top_right_lon - bottom_left_lon

    # Calculate the distances in meters
    lat_diff_m = lat_diff * 111139  # 1 degree latitude = 111139 m
    lon_diff_m = lon_diff * 111139 * np.cos(np.radians(
        # 1 degree longitude = 111139 m * cos(latitude)
        (top_right_lat + bottom_left_lat) / 2))

    # Calculate the cell size
    cell_width = lon_diff_m / size_x
    cell_height = lat_diff_m / size_y
    return cell_width, cell_height


def mg_per_m3_to_ppm(molecular_weight, mg_per_m3=1):
    """
    Convert mg/m³ to ppm (parts per million) using the molecular weight of the substance.
    Args:
        molecular_weight (float): Molecular weight of the substance in g/mol.
        mg_per_m3 (float): Value in mg/m³. -> default is 1 to only get the conversion factor
        
    Returns:
        float: Value in ppm.
    """
    molar_volume = 24.45  # Molar volume of an ideal gas (R*T/P) in L/mol -> at 25 °C and 1 atm (EPA AQI reference conditions)

    ppm = (mg_per_m3 * molar_volume) / (molecular_weight)  # Convert to ppm
    return ppm


def array_to_csv(array, csv_name, folder_path, verbose=False):
    """

    Save a numpy array to a CSV file.
    Args:
        array (numpy array): The array to save.
        csv_name (str): The name of the CSV file (without extension).
        folder_path (str): The path to the folder where the CSV file will be saved.
        verbose (bool): If True, prints additional information.
    Outputs:
        None (only saves to a file)
    """
    printv("Start function array_to_csv",
           verbose=verbose, color="blue", decorate=True)
    save_path = os.path.join(folder_path, csv_name + ".csv")
    df = pd.DataFrame(array)
    df.to_csv(save_path, index=False)
