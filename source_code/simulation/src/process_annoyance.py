import glob
import os

import numpy as np
import pandas as pd

from helper import printv, save_vec_to_csv, mg_per_m3_to_ppm

# Per pollutant: averaging window (file name pattern of the time averages written by
# process_data.py), the unit conversion from mg/m3 for the US AQI and for the AQIH, and the
# pollutant name used by the two AQI functions below.
POLLUTANT_CONFIG = {
    "co":  {"pattern": "_8h_avg_from_hour", "name": "CO",
            "conversion_factor_us_AQI": mg_per_m3_to_ppm(28.01),         # mg/m3 -> ppm (CO, 28.01 g/mol)
            "conversion_factor_AQIH": 1e3},
    "nox": {"pattern": "_1h_avg_hour", "name": "NO2",
            "conversion_factor_us_AQI": mg_per_m3_to_ppm(46.01) * 1000,  # mg/m3 -> ppb (NO2, 46.01 g/mol)
            "conversion_factor_AQIH": 1e3},                              # mg/m3 -> ug/m3
    "pmx": {"pattern": "_24h_avg_from_hour", "name": "PM2.5",
            "conversion_factor_us_AQI": 1e3,                             # mg/m3 -> ug/m3
            "conversion_factor_AQIH": 1e3},
}


def max_index_over_time(processed_data_folder, pollutant, conversion_factor, aqi_function, verbose=False):
    """
    Per-cell maximum over all time windows of `aqi_function(concentration * conversion_factor)`.
    Returns None if there are no time averages for the pollutant.
    """
    cfg = POLLUTANT_CONFIG[pollutant]
    files = glob.glob(os.path.join(processed_data_folder, f"data_{pollutant}{cfg['pattern']}*.csv"))
    if not files:
        printv(f"No files found for {pollutant} with pattern {cfg['pattern']}",
               verbose=verbose, color="yellow")
        return None

    index_of = np.vectorize(lambda c: aqi_function(c, cfg["name"]), otypes=[float])
    max_index = None
    for file_path in files:
        data = pd.read_csv(file_path).values
        window_index = index_of(data * conversion_factor)
        max_index = window_index if max_index is None else np.maximum(max_index, window_index)
        time_label = os.path.basename(file_path).split(cfg["pattern"])[1].split(".csv")[0]
        printv(f"Processed {pollutant} for time {time_label}", verbose=verbose)
    return max_index


def process_data_annoyance(data_folder, final_results_folder, processed_data_folder, verbose=False):
    """
    Calculate the maximum US AQI (CO, NO2, PM2.5) and AQIH (NO2, PM2.5) of every cell over the day,
    each from its own averaging window (8-hour CO, 1-hour NO2, 24-hour PM2.5), and the
    per-cell maximum over the pollutants ("overall").

    Args:
        data_folder: Path to folder containing the data files (unused, kept for the callers)
        final_results_folder: Path to folder where final results will be saved
        processed_data_folder: Path to the time averages written by process_data_statistics
        verbose: Whether to print verbose output
    """
    printv("Start function process_data_annoyance",
           verbose=verbose, color="cyan", decorate=True)

    systems = [
        # (output tag, pollutants, conversion factor key, AQI function)
        ("aqi",  ["co", "nox", "pmx"], "conversion_factor_us_AQI", calculate_aqi_for_pollutant_US_AQI),
        ("aqih", ["nox", "pmx"],       "conversion_factor_AQIH",   calculate_aqi_for_pollutant_AQIH),
    ]
    for tag, pollutants, factor_key, aqi_function in systems:
        max_by_pollutant = {}
        for pollutant in pollutants:
            printv(f"Processing {pollutant} {tag}...", verbose=verbose, color="cyan")
            max_index = max_index_over_time(processed_data_folder, pollutant,
                                            POLLUTANT_CONFIG[pollutant][factor_key], aqi_function, verbose)
            if max_index is None:
                continue
            filename = f"{pollutant}_max_{tag}.csv"
            save_vec_to_csv(max_index, final_results_folder, filename)
            printv(f"Saved maximum {pollutant} {tag} to {filename}", verbose=verbose, color="green")
            max_by_pollutant[pollutant] = max_index

        if max_by_pollutant:
            overall = np.maximum.reduce(list(max_by_pollutant.values()))
            save_vec_to_csv(overall, final_results_folder, f"overall_max_{tag}.csv")
            printv(f"Saved overall maximum {tag} to overall_max_{tag}.csv", verbose=verbose, color="green")
        else:
            printv(f"No {tag} data was calculated for any pollutant", verbose=verbose, color="yellow")

    printv("Finished processing data for annoyance",
           verbose=verbose, color="green", decorate=True)


def calculate_aqi_for_pollutant_US_AQI(concentration, pollutant):
    """
    Calculate AQI for a specific pollutant based on its concentration

    Args:
        concentration: Pollutant concentration (For CO: ppm, NO2: ppb, PM2.5: μg/m³)
        pollutant: Type of pollutant ('CO', 'NO2', or 'PM2.5')

    Returns:
        AQI value (0-500 scale)
    """
    # AQI breakpoints for different pollutants
    # Format: [concentration_low, concentration_high, index_low, index_high]

    # Source: CO/NO2 breakpoints from EPA's original AQI Technical Assistance Document
    # PM2.5 breakpoints updated per the February 2024 EPA revisions

    breakpoints = {
        'CO': [   # ppm, 8-h
            [0, 4.4, 0, 50],
            [4.5, 9.4, 51, 100],
            [9.5, 12.4, 101, 150],
            [12.5, 15.4, 151, 200],
            [15.5, 30.4, 201, 300],
            [30.5, 40.4, 301, 400],
            [40.5, 50.4, 401, 500]
        ],
        'NO2': [  # ppb, 1-h
            [0, 53, 0, 50],
            [54, 100, 51, 100],
            [101, 360, 101, 150],
            [361, 649, 151, 200],
            [650, 1249, 201, 300],
            [1250, 1649, 301, 400],
            [1650, 2049, 401, 500]
        ],
        'PM2.5': [  # ug/m3, 24-h;
            [0, 9.0, 0, 50],
            [9.1, 35.4, 51, 100],
            [35.5, 55.4, 101, 150],
            [55.5, 125.4, 151, 200],
            [125.5, 225.4, 201, 300],
            [225.5, 325.4, 301, 500]
        ]
    }

    bps = breakpoints[pollutant]
    if concentration <= 0:
        return 0
    if concentration > bps[-1][1]:
        return 500
    for i, bp in enumerate(bps):
        if concentration <= bp[1]:
            if concentration >= bp[0]:
                return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]
            # Concentration falls in the gap between the previous band and this one
            # (EPA truncates instead): interpolate across the gap.
            prev = bps[i - 1]
            return prev[3] + (bp[2] - prev[3]) * (concentration - prev[1]) / (bp[0] - prev[1])
    return 500

def calculate_aqi_for_pollutant_AQIH(concentration, pollutant):
    """
    Calculate AQI for a specific pollutant based on its concentration

    Args:
        concentration: Pollutant concentration (For NO2: μg/m³, PM2.5: μg/m³)
        pollutant: Type of pollutant ('NO2', or 'PM2.5')

    Returns:
        AQI value (1-10 scale)
    """
    # AQIH breakpoints for different pollutants
    # Format: [concentration_low, concentration_high, index_low, index_high]
    # Source: https://airquality.ie/information/air-quality-index-for-health
    
    breakpoints = {
        'NO2': [
            [0, 67, 1, 1],
            [67, 134, 1, 2],
            [134, 200, 2, 3],
            [200, 267, 3, 4],
            [267, 334, 4, 5],
            [334, 400, 5, 6],
            [400, 467, 6, 7], 
            [467, 534, 7, 8],
            [534, 600, 8, 9],
            [600, float('inf'), 9, 10]
        ],
        'PM2.5': [
            [0, 11, 1, 1],
            [11, 23, 1, 2],
            [23, 35, 2, 3],
            [35, 53, 3, 4],
            [53, 70, 4, 5],
            [70, 88, 5, 6],
            [88, 106, 6, 7],
            [106, 124, 7, 8],
            [124, 142, 8, 9],
            [142, float('inf'), 9, 10]
        ]
    }

    
    # If pollutant is CO, ignore it as per Irish AQIH
    if pollutant == 'CO':
        assert(False), "CO is not considered in the AQIH system"
    
    # If concentration is negative, return the lowest index
    if concentration < 0:
        printv("negative concentration in AQIH detected", True, "red")
        return 1
    
    # Find the appropriate breakpoint and apply linear interpolation
    for bp in breakpoints[pollutant]:
        if bp[0] <= concentration < bp[1]:
            if bp[1] == float('inf'):
                return bp[3]   # open-ended top band: the highest index (interpolating over an infinite width never leaves bp[2])
            # Linear interpolation
            aqi = ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]
            return aqi
    
    # If concentration is higher than the highest breakpoint
    return 10

