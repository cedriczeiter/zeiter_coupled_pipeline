"""
Download the base-map data for the study-area figure into data/study_area_basemap/.

Sources (all open data, credit in the figure caption / acknowledgements):
  - OpenStreetMap via the Overpass API: roads, railway, rivers, small water bodies,
    and building footprints around each simulation domain
  - Mapzen/AWS terrain tiles (terrarium encoding, zoom 13) for the hillshade
  - Natural Earth 1:10m: Swiss canton outlines and lakes (Switzerland inset),
    trimmed to Switzerland before saving

Existing files are skipped, so re-running only fetches what is missing. The
snapshot used for the paper is already in data/, so this script is only needed
to rebuild it.

Usage:
  python fetch_study_area_basemap.py
  OVERPASS=https://overpass.private.coffee/api/interpreter python fetch_study_area_basemap.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import math
import time
import urllib.parse
import urllib.request

from pyproj import Transformer

from _shared import DATA_DIR, VILLAGES

OUT_DIR = os.path.join(DATA_DIR, "study_area_basemap")
OVERPASS = os.environ.get("OVERPASS", "https://overpass-api.de/api/interpreter")
HEADERS = {"User-Agent": "FDM-SUMO-study-area-figure/1.0"}
NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"

S, W, N, E = 46.645, 8.545, 46.915, 8.725   # overview extent (lat/lon), Göschenen to Flüelen
NET_OFFSET = (455283.56, 5149384.03)        # SUMO netOffset of all six village networks (UTM 32N)
BUILDING_HALF_WINDOW = (1500, 2600)         # [m] around each domain centre; covers every detail panel
DEM_ZOOM = 13


def fetch(url, path, query=None):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    body = urllib.parse.urlencode({"data": query}).encode() if query else None
    req = urllib.request.Request(url, data=body, headers=HEADERS)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            print(f"  ok   {os.path.relpath(path, DATA_DIR)} ({len(data) / 1e6:.1f} MB)")
            return
        except Exception as e:  # noqa: BLE001  (network errors of any kind -> retry)
            print(f"  retry {os.path.basename(path)}: {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def building_windows():
    """Lat/lon boxes around each simulation domain for the building query."""
    to_ll = Transformer.from_crs(32632, 4326, always_xy=True)
    ox, oy = NET_OFFSET
    hw, hh = BUILDING_HALF_WINDOW
    boxes = []
    for v in VILLAGES.values():
        cx = (v["GL"] + v["GR"]) / 2 + ox
        cy = (v["GB"] + v["GT"]) / 2 + oy
        w, s = to_ll.transform(cx - hw, cy - hh)
        e, n = to_ll.transform(cx + hw, cy + hh)
        boxes.append((s, w, n, e))
    return boxes


def fetch_osm():
    bb = f"({S},{W},{N},{E})"
    fetch(OVERPASS, os.path.join(OUT_DIR, "roads_rail_rivers.json"), f"""
[out:json][timeout:240];
(
  way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|unclassified|residential|living_street)$"]{bb};
  way["railway"="rail"]{bb};
  way["waterway"="river"]{bb};
);
out tags geom;
""")
    fetch(OVERPASS, os.path.join(OUT_DIR, "water.json"), f"""
[out:json][timeout:240];
(
  way["natural"="water"]{bb};
  relation["natural"="water"]{bb};
);
out tags geom;
""")
    parts = "".join(f'way["building"]({s},{w},{n},{e});' for s, w, n, e in building_windows())
    fetch(OVERPASS, os.path.join(OUT_DIR, "buildings.json"), f"""
[out:json][timeout:240];
({parts});
out geom;
""")


def fetch_natural_earth():
    """Download Natural Earth layers and keep only what the Switzerland inset needs."""
    import geopandas as gpd
    import pandas as pd

    cantons = os.path.join(OUT_DIR, "ch_cantons.geojson")
    lakes = os.path.join(OUT_DIR, "ch_lakes.geojson")
    if os.path.exists(cantons) and os.path.exists(lakes):
        return
    tmp = os.path.join(OUT_DIR, "_ne_tmp")
    os.makedirs(tmp, exist_ok=True)
    for f in ("ne_10m_admin_1_states_provinces.geojson", "ne_10m_lakes.geojson", "ne_10m_lakes_europe.geojson"):
        fetch(NE_URL + f, os.path.join(tmp, f))
    adm = gpd.read_file(os.path.join(tmp, "ne_10m_admin_1_states_provinces.geojson"))
    ch = adm[adm["adm0_a3"] == "CHE"][["name", "geometry"]]
    ch.to_file(cantons, driver="GeoJSON")
    lk = pd.concat([gpd.read_file(os.path.join(tmp, f))[["name", "geometry"]]
                    for f in ("ne_10m_lakes.geojson", "ne_10m_lakes_europe.geojson")])
    lk = lk[lk.intersects(ch.union_all())]
    gpd.GeoDataFrame(lk, crs=adm.crs).to_file(lakes, driver="GeoJSON")
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("  ok   ch_cantons.geojson, ch_lakes.geojson")


def fetch_dem():
    def tile(lat, lon):
        n = 2 ** DEM_ZOOM
        return (int((lon + 180) / 360 * n),
                int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))
    x0, y0 = tile(N, W)
    x1, y1 = tile(S, E)
    dem_dir = os.path.join(OUT_DIR, "dem")
    os.makedirs(dem_dir, exist_ok=True)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            fetch(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{DEM_ZOOM}/{x}/{y}.png",
                  os.path.join(dem_dir, f"{DEM_ZOOM}_{x}_{y}.png"))
    with open(os.path.join(dem_dir, "index.json"), "w") as f:
        json.dump(dict(z=DEM_ZOOM, x0=x0, x1=x1, y0=y0, y1=y1), f)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching study-area base map into {OUT_DIR}")
    fetch_osm()
    fetch_natural_earth()
    fetch_dem()
    print("done")
