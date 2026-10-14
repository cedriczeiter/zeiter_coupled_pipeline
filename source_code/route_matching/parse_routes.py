"""
Reconstruct trips from the detector files of one date: a vehicle that was seen entering
the village (an `entry` sensor) and leaving it (an `exit` sensor) gets one trip.

    python parse_routes.py [--sensor-dir data/czeiter_loop_output] [--out-dir output]

Reads  <sensor-dir>/<village>/<village>_{entry,exit}_*.xml
Writes <out-dir>/reconstructed_trips_<village>.csv
"""
import argparse
import csv
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

VILLAGES = ['erstfeld', 'goeschenen', 'gurtnellen', 'schattdorf', 'silenen', 'wassen']


def parse_sensor_files(sensor_dir, village):
    """Entry / exit event of every vehicle seen by the sensors of one village; returns (vehicles, n_events)."""
    vehicles = defaultdict(lambda: {"entry": None, "exit": None})
    total_datapoints = 0
    for filename in sorted(os.listdir(sensor_dir)):
        if not (filename.startswith(village) and filename.endswith(".xml")):
            print(f"Skipping {filename}")
            continue
        is_entry = "entry" in filename
        is_exit = "exit" in filename
        if not (is_entry or is_exit):
            print(f"Skipping {filename} (not an entry or exit file)")
            continue
        print(f"Parsing {filename}...")

        # entry sensors: a vehicle entering is the start of its trip; exit sensors: leaving is the end
        wanted_state = "enter" if is_entry else "leave"
        kind = "entry" if is_entry else "exit"
        root = ET.parse(os.path.join(sensor_dir, filename)).getroot()
        for elem in root.findall("instantOut"):   # the output of the sensors
            total_datapoints += 1
            if elem.get("state") != wanted_state:
                continue
            # e.g. 'erstfeld_entry_A' -> 'lane_entry_A'
            lane_id = elem.get("id").replace(f"{village}_{kind}", f"lane_{kind}")
            vehicles[elem.get("vehID")][kind] = {
                "time": float(elem.get("time")),   # seconds
                "edge": lane_id,
                "type": elem.get("type"),
                "speed": float(elem.get("speed")),
            }
    return vehicles, total_datapoints


def complete_trips(vehicles):
    """Vehicles with both an entry and an exit event."""
    return [{
        "vehID": vehID,
        "depart": data["entry"]["time"],
        "from": data["entry"]["edge"],
        "to": data["exit"]["edge"],
        "arrival": data["exit"]["time"],
        "arrival_speed": data["exit"]["speed"],
        "type": data["entry"]["type"],
    } for vehID, data in vehicles.items() if data["entry"] and data["exit"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor-dir", default="data/czeiter_loop_output")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for village in VILLAGES:
        print("-" * 49)
        vehicles, total_datapoints = parse_sensor_files(os.path.join(args.sensor_dir, village), village)
        trips = complete_trips(vehicles)

        output_csv = os.path.join(args.out_dir, f"reconstructed_trips_{village}.csv")
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "vehID", "depart", "from", "to", "arrival", "arrival_speed", "type"])
            writer.writeheader()
            writer.writerows(trips)

        # every sensor logs enter and leave, and there are entry and exit sensors: expect >= 4 events per trip
        print(f"Saved {len(trips)} trips to {output_csv} from {total_datapoints} sensor events")


if __name__ == "__main__":
    main()
