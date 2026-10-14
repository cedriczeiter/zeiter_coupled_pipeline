import ast
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd

import config as _cfg_pl
from emission_models import Z_THICKNESSES
from helper import printv, array_to_csv


def _polygon_coords(sumo_polygon, sumo_bbox_polygon):
    """Outline of a house from the converted CSV: the OSM polygon, else its bounding box, else None."""
    for candidate in (sumo_polygon, sumo_bbox_polygon):
        if isinstance(candidate, str) and len(candidate) > 2:
            try:
                coords = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
            if coords:
                return coords
    return None


def plot_heatmap(input_vec, only_houses, output_file_name, GRID_LEFT, GRID_RIGHT, GRID_BOTTOM, GRID_TOP, GRID_DIM_X, GRID_DIM_Y, verbose=False, house_csv_name=None, house_coloring=False, output_path="../output/plot/", vmin=None, vmax=None, line_color="green"):
    """
    Plots a heatmap of the input array and overlays the house outlines from the converted house CSV.

    Args:
        input_vec (numpy 2D array): 2D array (GRID_DIM_X, GRID_DIM_Y) representing the heatmap data.
        only_houses (bool): Make the heatmap itself transparent (only the houses are visible)
        output_file_name (str): Name of the output file (without extension)
        GRID_LEFT, GRID_RIGHT, GRID_BOTTOM, GRID_TOP (float): grid bounds in SUMO coordinates
        GRID_DIM_X, GRID_DIM_Y (int): number of grid cells
        verbose (bool): If True, prints additional information.
        house_csv_name (str): csv file where houses are stored (SUMO coordinates)
        house_coloring (bool): Color each house by the value of the grid cell it lies in
        output_path (str): In which folder the plot should be saved to
        vmin, vmax (float): Color scale limits
        line_color (str): Color of the house outlines
    Outputs:
        None (only saves to a file)
    """

    plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
    })

    printv("Start function plot_heatmap",
           verbose=verbose, color="blue", decorate=True)

    plt.figure(figsize=(10, 8))

    hm = plt.imshow(input_vec.T, cmap='hot_r', interpolation='nearest',
                    extent=[GRID_LEFT, GRID_RIGHT, GRID_BOTTOM, GRID_TOP],
                    origin='lower', vmin=vmin, vmax=vmax, alpha=0 if only_houses else 1)

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.axis('off')
    plt.xlim(GRID_LEFT, GRID_RIGHT)
    plt.ylim(GRID_BOTTOM, GRID_TOP)
    hm_colors = hm.cmap(hm.norm(hm.get_array()))   # RGBA of every cell, indexed [y, x]

    if house_csv_name is not None:
        try:
            house_polygons_df = pd.read_csv(house_csv_name)
            printv(f"Loaded {len(house_polygons_df)} house polygons from {house_csv_name}", verbose=verbose)

            cell_width = (GRID_RIGHT - GRID_LEFT) / GRID_DIM_X
            cell_height = (GRID_TOP - GRID_BOTTOM) / GRID_DIM_Y

            for i, row in house_polygons_df.iterrows():
                try:
                    coords = _polygon_coords(row.get('sumo_polygon'), row.get('sumo_bbox_polygon'))
                    if not coords:
                        continue
                    if house_coloring:
                        house_grid_x = int((row.get('sumo_x') - GRID_LEFT) / cell_width)
                        house_grid_y = int((row.get('sumo_y') - GRID_BOTTOM) / cell_height)
                        if 0 <= house_grid_y < hm_colors.shape[0] and 0 <= house_grid_x < hm_colors.shape[1]:
                            color = hm_colors[house_grid_y, house_grid_x]
                        else:
                            color = np.array([0.5, 0.5, 0.5, 1.0])  # house outside the grid
                        ax.add_patch(MplPolygon(coords, closed=True, edgecolor=line_color,
                                                facecolor=color, linewidth=0, alpha=1.0))
                    else:
                        ax.add_patch(MplPolygon(coords, closed=True, edgecolor=line_color,
                                                facecolor="none", linewidth=0.5, alpha=1.0))
                except Exception as e:
                    printv(f"Error processing house polygon at index {i}: {e}", verbose=True, color="Red")

        except Exception as e:
            printv(f"Error loading house CSV {house_csv_name}: {e}", verbose=True, color="Red")

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path+f'/{output_file_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path+f'/{output_file_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    printv("Finished the plot", verbose=verbose)


def plot_all_heatmaps(data_folder, plot_folder, house_csv_name=None, print_houses=False, verbose=False):
    """
    Plots a heatmap for every CSV in data_folder (grid taken from config).

    Args:
        data_folder (str): Where to find all the csv
        plot_folder (str): Where to save the plots to
        house_csv_name (str): Converted house polygon CSV (SUMO coordinates); None = no houses
        print_houses (bool): Outline the houses on every plot (needs house_csv_name)
        verbose (bool): If True, prints additional information.
    """
    printv("Start function plot_all_heatmaps",
           verbose=verbose, color="blue", decorate=True)

    os.makedirs(plot_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    printv(f"Found {len(csv_files)} CSV files in {data_folder}", verbose=verbose)

    grid = dict(GRID_LEFT=_cfg_pl.GRID_LEFT, GRID_RIGHT=_cfg_pl.GRID_RIGHT,
                GRID_BOTTOM=_cfg_pl.GRID_BOTTOM, GRID_TOP=_cfg_pl.GRID_TOP,
                GRID_DIM_X=_cfg_pl.GRID_DIM_X, GRID_DIM_Y=_cfg_pl.GRID_DIM_Y)
    houses = house_csv_name if print_houses else None

    for csv_file in csv_files:
        try:
            base_name = os.path.basename(csv_file).replace(".csv", "")
            printv(f"Processing {base_name}", verbose=verbose)
            array_data = pd.read_csv(csv_file).values

            plot_name = f"{base_name}_mg_per_m3" if base_name.startswith("data_") else base_name

            plot_heatmap(array_data, only_houses=False, output_file_name=plot_name, verbose=verbose,
                         house_csv_name=houses, output_path=plot_folder, **grid)

            # For annoyance data, also create a version with house coloring
            if "annoyance" in base_name.lower() and house_csv_name is not None:
                clipped_data = np.copy(array_data)
                clipped_data[clipped_data > 10000] = 0
                plot_heatmap(clipped_data, only_houses=False, output_file_name=f"{plot_name}_house_colored",
                             verbose=verbose, house_csv_name=house_csv_name, house_coloring=True,
                             output_path=plot_folder, **grid)

        except Exception as e:
            printv(f"Error processing {csv_file}: {e}", verbose=True, color="Red")


def save_all_data(co_vector_global, nox_vector_global, pmx_vector_global,
                  cell_len_x_m, cell_len_y_m,
                  folder_path="../output/data/", verbose=False, identifier="default"):
    """
    Save a concentration snapshot [mg/m^3] of every z-layer as separate CSVs.

    Naming: data_<pollutant>_<layer>_<identifier>.csv
      layer 0 = injection layer (0-0.3 m)
      layer 1 = exposure layer (0.3-1.2 m), used for AQI and plots (config.EXPOSURE_LAYER)
      layers 2-5 = upper geometric layers, saved for further analysis

    The fields are masses per cell [mg]; layer thicknesses come from emission_models.Z_THICKNESSES.
    """
    printv("Start function save_all_data", verbose=verbose, color="blue", decorate=True)

    os.makedirs(folder_path, exist_ok=True)

    NZ = co_vector_global.shape[2]
    for k in range(NZ):
        scale = 1.0 / (cell_len_x_m * cell_len_y_m * Z_THICKNESSES[k])
        array_to_csv(co_vector_global[:, :, k]  * scale, f"data_co_{k}_{identifier}",  folder_path)
        array_to_csv(nox_vector_global[:, :, k] * scale, f"data_nox_{k}_{identifier}", folder_path)
        array_to_csv(pmx_vector_global[:, :, k] * scale, f"data_pmx_{k}_{identifier}", folder_path)
