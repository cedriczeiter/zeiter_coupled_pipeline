"""
Per-second speed of every exit lane over one day, from the vehicles' arrival speeds:
the speed of the last vehicle that left, held until the next one (-1 before the first).
simulation/src/sumo_commands.py:update_edge_speeds() applies this as the speed limit of the exit lanes.

    python speed_regulation.py [--date 20_11] [--out-dir output]

Reads  <out-dir>/reconstructed_trips_<village>.csv  (from parse_routes.py)
Writes <out-dir>/<date>_exit_speeds_per_second_<village>.csv
"""
import argparse
import os

import pandas as pd

VILLAGES = ['erstfeld', 'goeschenen', 'gurtnellen', 'schattdorf', 'silenen', 'wassen']
END_TIME = 3600 * 24   # full day of speeds [s]
NO_DATA = -1

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--date", default="20_11", help="MM_DD prefix of the output files")
parser.add_argument("--out-dir", default="output")
args = parser.parse_args()

for village in VILLAGES:
    df = pd.read_csv(os.path.join(args.out_dir, f"reconstructed_trips_{village}.csv"))
    df_exit = df[["arrival", "to", "arrival_speed"]].rename(
        columns={"to": "exit_edge", "arrival": "time", "arrival_speed": "speed"})

    exit_edges = sorted(df_exit["exit_edge"].unique())
    exit_speed_map = {edge: [NO_DATA] * (END_TIME + 1) for edge in exit_edges}

    # record the speed at the arrival time of each vehicle
    for _, row in df_exit.iterrows():
        t = int(row["time"])
        if 0 <= t <= END_TIME:
            exit_speed_map[row["exit_edge"]][t] = row["speed"]

    # hold the last known speed until the next arrival
    for edge in exit_edges:
        last_speed = NO_DATA
        for t, speed in enumerate(exit_speed_map[edge]):
            if speed != NO_DATA:
                last_speed = speed
            elif last_speed != NO_DATA:
                exit_speed_map[edge][t] = last_speed

    result_df = pd.DataFrame({"time": range(END_TIME + 1), **exit_speed_map})
    output_csv = os.path.join(args.out_dir, f"{args.date}_exit_speeds_per_second_{village}.csv")
    result_df.to_csv(output_csv, index=False)
    print(output_csv)
