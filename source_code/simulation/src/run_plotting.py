"""
Generate paper-style heatmap plots from a simulation output folder.

Usage (from simulation/src/):
    python run_plotting.py <output_folder> [--times 32400 64800] [--fill]

    <output_folder>  folder written by run_sim.py: a name relative to simulation/ (where run_sim.py
                     writes its output_* folders), e.g.
                     output_goeschenen_07_20_rt_True_rp_0.05_wd_False_2026_06_01_12_00_00,
                     or an absolute path.
    --times          space-separated list of snapshot times to plot (seconds).
                     Default: all available times.
    --fill           also produce a "filled-houses" version of each plot.

Output goes into a "paper_plots/" subfolder inside <output_folder>.

The folder name encodes all parameters, following the convention:
    output_<village>_<date>_rt_<random>_rp_<reroute>_wd_<weekday>_<timestamp>
Only the two lowest z-layers (0-0.3 m and 0.3-1.2 m) are plotted.
"""

import argparse
import ast
import glob
import os
import re
import sys

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from config import VILLAGE_CONFIGS
from helper import convert_lat_lon_to_sumo_coordinates
from plotting import plot_heatmap

# colour scale limits [mg/m3] used for the paper's heatmaps
VMAX = {"co": 0.27, "nox": 0.07, "pmx": 0.0065}
GAS_LABELS = {"co": "CO", "nox": "NO2", "pmx": "PM2.5"}

FOLDER_PATTERN = re.compile(r"output_(?P<village>[a-z]+)_(?P<date>\d\d_\d\d)_rt_(?P<rt>True|False)"
                            r"_rp_(?P<rp>[\d.]+)_wd_(?P<wd>True|False)_")


def parse_folder_name(folder_name):
    """Extract (village, date, random_traffic, reroute_pct, weekday) from folder name."""
    m = FOLDER_PATTERN.match(folder_name)
    if m is None:
        raise ValueError(f"not a run_sim.py output folder name: {folder_name}")
    return m["village"], m["date"], m["rt"] == "True", float(m["rp"]), m["wd"] == "True"


def _convert_house_polygons_to_sumo(src_csv, out_csv, netfile):
    """
    Read the house_polygons CSV (lat/lon) and write a new CSV with
    SUMO-coordinate polygon columns.  Result is cached at out_csv.
    """
    df = pd.read_csv(src_csv)

    df["sumo_x"], df["sumo_y"] = zip(*df.apply(
        lambda r: convert_lat_lon_to_sumo_coordinates(r["lat"], r["lon"], netfile),
        axis=1,
    ))

    def bbox_to_poly(bbox_str):
        if not isinstance(bbox_str, str):
            return []
        try:
            min_lat, max_lat, min_lon, max_lon = map(float, ast.literal_eval(bbox_str))
            corners = [(min_lon, min_lat), (min_lon, max_lat),
                       (max_lon, max_lat), (max_lon, min_lat)]
            return [convert_lat_lon_to_sumo_coordinates(lat, lon, netfile)
                    for lon, lat in corners]
        except Exception:
            return []

    def parse_poly(poly_str):
        if not isinstance(poly_str, str):
            return []
        try:
            if poly_str.startswith("POLYGON"):
                inner = poly_str[9:-2]
                pairs = [p.strip() for p in inner.split(",")]
                return [convert_lat_lon_to_sumo_coordinates(float(lat), float(lon), netfile)
                        for lon, lat in [p.split() for p in pairs]]
        except Exception:
            pass
        return []

    df["sumo_bbox_polygon"] = df["bounding_box"].apply(bbox_to_poly)
    df["sumo_polygon"]      = df["polygon"].apply(parse_poly)

    df[["lat", "lon", "sumo_x", "sumo_y", "sumo_bbox_polygon", "sumo_polygon"]].to_csv(out_csv, index=False)
    print(f"  Converted house polygons saved to {out_csv}")


def make_plots(output_path, village, date, rt, rp, wd, times_filter=None, fill=False, verbose=False):
    cfg = VILLAGE_CONFIGS[village]
    GL, GR = cfg["GRID_LEFT"], cfg["GRID_RIGHT"]
    GB, GT = cfg["GRID_BOTTOM"], cfg["GRID_TOP"]
    NX, NY = cfg["GRID_DIM_X"], cfg["GRID_DIM_Y"]
    net_file = os.path.join(SRC_DIR, f"../villages/{cfg['VILLAGE_LARGE']}/{village}_osm.net.xml")

    data_folder  = os.path.join(output_path, "data")
    plot_folder  = os.path.join(output_path, "paper_plots")
    os.makedirs(plot_folder, exist_ok=True)

    # House polygon CSV (converted to SUMO coords)
    temp_dir  = os.path.join(SRC_DIR, "../output/temp")
    src_csv   = os.path.join(temp_dir, f"house_polygons_{GL}_{GR}_{GB}_{GT}.csv")
    house_csv = os.path.join(temp_dir, f"converted_house_polygons_{GL}_{GR}_{GB}_{GT}.csv")
    if not os.path.exists(house_csv):
        if os.path.exists(src_csv):
            print("  Converting house polygons to SUMO coordinates ...")
            _convert_house_polygons_to_sumo(src_csv, house_csv, net_file)
        else:
            house_csv = None
            print("  No house polygon file found — plotting without house overlay.")

    csv_files = glob.glob(os.path.join(data_folder, "data_*.csv"))
    print(f"  Found {len(csv_files)} snapshot CSVs in {data_folder}")

    for csv_file in sorted(csv_files):
        # Expected format: data_<gas>_<layer>_<time[.frac]>
        parts = os.path.basename(csv_file).replace(".csv", "").split("_")
        if len(parts) < 4 or parts[1] not in GAS_LABELS:
            continue
        gas_key = parts[1]
        if int(parts[2]) >= 2:   # only the first 2 layers, to save space
            continue

        time_str = parts[3].split(".")[0]   # strip fractional seconds
        if times_filter and time_str not in times_filter:
            continue

        array_data = pd.read_csv(csv_file).values
        base_name  = (f"{village}_{date}_{GAS_LABELS[gas_key]}_{time_str}"
                      f"___random_traffic_{rt}"
                      f"___reroute_percentage_{rp}"
                      f"___weekday_{wd}")
        common = dict(input_vec=array_data, GRID_LEFT=GL, GRID_RIGHT=GR, GRID_BOTTOM=GB, GRID_TOP=GT,
                      GRID_DIM_X=NX, GRID_DIM_Y=NY, verbose=verbose, house_csv_name=house_csv,
                      vmax=VMAX[gas_key], output_path=plot_folder)

        plot_heatmap(only_houses=False, output_file_name=base_name, house_coloring=False, **common)
        if fill:
            plot_heatmap(only_houses=True, output_file_name=f"fill_{base_name}", house_coloring=True, **common)

        print(f"    Saved: {base_name}.png/pdf")

    print(f"  All plots written to {plot_folder}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_folder",
                        help="Folder name (relative to simulation/) or absolute path")
    parser.add_argument("--times", nargs="*",
                        help="Snapshot times to plot (seconds). Default: all.")
    parser.add_argument("--fill", action="store_true",
                        help="Also produce filled-houses versions.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_path = os.path.join(SRC_DIR, "..", args.output_folder)   # absolute paths stay as they are
    if not os.path.isdir(output_path):
        sys.exit(f"Error: folder not found: {output_path}")

    village, date, rt, rp, wd = parse_folder_name(os.path.basename(os.path.normpath(output_path)))
    print(f"Village: {village}  Date: {date}  rt={rt}  rp={rp}  wd={wd}")

    make_plots(
        output_path=output_path,
        village=village, date=date, rt=rt, rp=rp, wd=wd,
        times_filter=set(args.times) if args.times else None,
        fill=args.fill,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
