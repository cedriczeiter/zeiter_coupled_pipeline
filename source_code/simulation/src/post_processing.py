import os

from plotting import plot_all_heatmaps
from process_annoyance import process_data_annoyance
from process_data import process_data_statistics


def post_processing_wrapper(data_folder, plot_folder, avg_data_folder, final_results_folder, verbose=False):
    """
    Post-process the data from the simulation: time averages, AQI per cell, heatmap plots.
    Args:
        data_folder (str): Path to the folder containing the simulation data.
        plot_folder (str): Path to the folder where plots will be saved.
        avg_data_folder (str): Path to the folder where average data will be saved.
        final_results_folder (str): Path to the folder where final results will be saved.
        verbose (bool): If more information should be printed.
    """
    for folder in (data_folder, plot_folder, avg_data_folder, final_results_folder):
        os.makedirs(folder, exist_ok=True)

    process_data_statistics(data_folder, avg_data_folder=avg_data_folder, verbose=verbose)

    process_data_annoyance(data_folder=data_folder, final_results_folder=final_results_folder,
                           processed_data_folder=avg_data_folder, verbose=verbose)

    plot_all_heatmaps(data_folder=final_results_folder, plot_folder=plot_folder, verbose=verbose)
    plot_all_heatmaps(data_folder=data_folder, plot_folder=plot_folder, verbose=verbose)
