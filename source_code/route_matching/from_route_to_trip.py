"""
Convert the reconstructed trips (CSV) to SUMO trip files, ordered by departure time.

    python from_route_to_trip.py [--out-dir output]
"""
import argparse
import os
import xml.etree.ElementTree as ET

import pandas as pd

VILLAGES = ['erstfeld', 'goeschenen', 'gurtnellen', 'schattdorf', 'silenen', 'wassen']

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", default="output")
args = parser.parse_args()

for village in VILLAGES:
    csv_file = os.path.join(args.out_dir, f"reconstructed_trips_{village}.csv")
    output_xml = os.path.join(args.out_dir, f"reconstructed_trips_{village}.trips.xml")

    df = pd.read_csv(csv_file)
    df.sort_values(by="depart", inplace=True)

    root = ET.Element("trips")
    for _, row in df.iterrows():
        trip = ET.SubElement(root, "trip")
        trip.set("id", row["vehID"])
        trip.set("depart", str(row["depart"]))
        trip.set("from", row["from"])
        trip.set("to", row["to"])
        trip.set("type", row["type"])

    ET.ElementTree(root).write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"SUMO trip file written to {output_xml}")
