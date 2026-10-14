"""
Generates a synthetic "resident population" CSV with the exact same schema
as the real (unpublishable) SynPop dataset.

Nothing here is derived from real data. Just for illustrative purposes.

Usage:
    python generate_fake_population.py [n_people] [output_csv]
"""
import os
import sys
import random

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_SRC = os.path.join(SRC_DIR, "..", "source_code", "simulation", "src")
sys.path.insert(0, REAL_SRC)
import sumolib.net as sumonet

NETFILE = os.path.join(SRC_DIR, "village_data", "Goeschenen", "goeschenen_osm.net.xml")

# Roughly plausible category proportions
AGE = ["0-17", "18-24", "25-44", "45-64", "65-74", "75+"]
AGE_W = [0.20, 0.05, 0.25, 0.30, 0.10, 0.10]

EMPLOYMENT = ["0", "1-39", "40-79", "80-100"]
EMPLOYMENT_W = [0.50, 0.10, 0.10, 0.30]

INCOME = ["0-50k", "50-95k", "95-145k", "145+"]
INCOME_W = [0.15, 0.40, 0.25, 0.20]

EDU = ["", "pupil", "apprentice", "student"]
EDU_W = [0.82, 0.13, 0.03, 0.02]

JOB = ["", "employee", "management", "ceo"]
JOB_W = [0.50, 0.35, 0.10, 0.05]

CAR_TYPE = ["", "BenzinKlein", "BenzinMittel", "DieselMittel", "BenzinGross"]
CAR_TYPE_W = [0.1, 0.2, 0.3, 0.3, 0.1]

PUBLIC_TRANSPORT = ["", "HTA", "GA", "VA"]
PT_W = [0.50, 0.40, 0.05, 0.05]


def _weighted(values, weights):
    return random.choices(values, weights=weights, k=1)[0]


def generate(n_people=500, seed=0):
    random.seed(seed)
    network = sumonet.readNet(NETFILE)
    xmin, ymin, xmax, ymax = network.getBoundary()

    rows = []
    tries = 0
    while len(rows) < n_people and tries < n_people * 20:
        tries += 1
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        # Keep only points SUMO can map back to a valid lon/lat
        try:
            lon, lat = network.convertXY2LonLat(x, y)
        except Exception:
            continue

        edu = _weighted(EDU, EDU_W)
        job = "" if edu in ("pupil", "student") else _weighted(JOB, JOB_W)
        employment = "0" if not job else _weighted(EMPLOYMENT[1:], EMPLOYMENT_W[1:])
        has_car = random.random() < 0.68

        rows.append({
            "age": _weighted(AGE, AGE_W),
            "level_of_employment": employment,
            "household_income": _weighted(INCOME, INCOME_W),
            "position_in_edu": edu,
            "position_in_bus": job,
            "cars_drivetype": _weighted(CAR_TYPE, CAR_TYPE_W) if has_car else "",
            "public_transport": _weighted(PUBLIC_TRANSPORT, PT_W),
            "xcoord": x,
            "ycoord": y,
            "lat": lat,
            "lon": lon,
            "car_and_license": "True" if has_car else "False",
        })

    return rows


COLUMNS = ["age", "level_of_employment", "household_income", "position_in_edu",
           "position_in_bus", "cars_drivetype", "public_transport",
           "xcoord", "ycoord", "lat", "lon", "car_and_license"]


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in COLUMNS) + "\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SRC_DIR, "fake_population.csv")
    rows = generate(n)
    write_csv(rows, out)
    print(f"Wrote {len(rows)} synthetic residents to {out}")
