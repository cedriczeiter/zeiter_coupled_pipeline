"""
Study area: relief overview of the Uri valley with the six simulation
domains (A-F), a Switzerland locator inset, and one detail panel per village.

Base-map data (OSM, terrain tiles, Natural Earth) are read from
data/study_area_basemap/; rebuild them with fetch_study_area_basemap.py.

Output:
  reproducibility/plotting/figures/study_area/study_area.pdf/png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import map_coordinates
import geopandas as gpd

from _shared import DATA_DIR, HOUSE_POLY_DIR, VILLAGES, VILLAGE_LABELS, VILLAGE_ORDER, save_fig

BASEMAP = os.path.join(DATA_DIR, "study_area_basemap")
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

# ------------------------------------------------------------------ parameters
P = dict(
    W_CM=16.0,
    gap_in=0.05,
    panel_aspect=0.62,          # width / height of each detail panel
    dom_frac=0.82,              # domain box height as a fraction of panel height
    ov_relief=(0.76, 0.24),     # base, amplitude of hillshade in the overview
    in_relief=0.35,             # contrast of hillshade in the panels (0..1)
    ov_res=25, in_px=420,
    swiss_rect=[0.0, 0.858, 0.55, 0.142],
)

ORANGE_FILL, ORANGE_EDGE, ORANGE_DARK = "#fbd3a6", "#e07b28", "#b85a12"
WATER, WATER_LINE = "#cfe0ec", "#a9c4d8"

# overview / panel line styles: (lw, colour, zorder)
OV_ROADS = {"motorway": (0.85, "#5c5c5c", 6), "trunk": (0.6, "#7a7a7a", 5), "primary": (0.6, "#7a7a7a", 5),
            "secondary": (0.4, "#8a8a8a", 4), "tertiary": (0.3, "#959595", 4)}
IN_ROADS = {"motorway": (1.3, "#6a6a6a", 6), "trunk": (1.0, "#7d7d7d", 5), "primary": (1.0, "#7d7d7d", 5),
            "secondary": (0.7, "#8c8c8c", 4), "tertiary": (0.6, "#8c8c8c", 4), "minor": (0.35, "#a3a3a3", 3)}

# ------------------------------------------------------------------ geometry
OX, OY = 455283.56, 5149384.03        # SUMO netOffset (shared by all six networks), UTM 32N
TO_UTM = Transformer.from_crs(4326, 32632, always_xy=True)
TO_LL = Transformer.from_crs(32632, 4326, always_xy=True)
UTM_TO_LV95 = Transformer.from_crs(32632, 2056, always_xy=True)

LABEL = VILLAGE_LABELS
ORDER = VILLAGE_ORDER
LETTER = {v: "ABCDEF"[i] for i, v in enumerate(ORDER)}
DOM = {v: (c["GL"] + OX, c["GR"] + OX, c["GB"] + OY, c["GT"] + OY) for v, c in VILLAGES.items()}

xs = [d[i] for d in DOM.values() for i in (0, 1)]
ys = [d[i] for d in DOM.values() for i in (2, 3)]
_, y_top = TO_UTM.transform(8.6239, 46.9047)       # Flüelen
OV = (min(xs) - 1800, max(xs) + 1800, min(ys) - 1300, y_top + 400)

# ------------------------------------------------------------------ data
def load_ways(path):
    out = []
    for e in json.load(open(path))["elements"]:
        if e["type"] != "way" or "geometry" not in e:
            continue
        x, y = TO_UTM.transform([g["lon"] for g in e["geometry"]], [g["lat"] for g in e["geometry"]])
        out.append((e.get("tags", {}), np.column_stack([x, y])))
    return out


WAYS = load_ways(os.path.join(BASEMAP, "roads_rail_rivers.json"))
WATERS = [xy for _, xy in load_ways(os.path.join(BASEMAP, "water.json")) if len(xy) > 3 and np.allclose(xy[0], xy[-1])]
BUILDINGS = [xy for _, xy in load_ways(os.path.join(BASEMAP, "buildings.json"))]
B_FIRST = np.array([b[0] for b in BUILDINGS])

HOUSES = {}
for v, c in VILLAGES.items():
    df = pd.read_csv(os.path.join(HOUSE_POLY_DIR,
                                  f"converted_house_polygons_{c['GL']}_{c['GR']}_{c['GB']}_{c['GT']}.csv"))
    HOUSES[v] = [np.array(ast.literal_eval(s)) + [OX, OY] for s in df["sumo_polygon"]]

idx = json.load(open(os.path.join(BASEMAP, "dem", "index.json")))
DZ, DX0, DY0 = idx["z"], idx["x0"], idx["y0"]
DEM = np.zeros(((idx["y1"] - DY0 + 1) * 256, (idx["x1"] - DX0 + 1) * 256))
for x in range(DX0, idx["x1"] + 1):
    for y in range(DY0, idx["y1"] + 1):
        a = np.asarray(Image.open(os.path.join(BASEMAP, "dem", f"{DZ}_{x}_{y}.png")).convert("RGB"), float)
        DEM[(y - DY0) * 256:(y - DY0 + 1) * 256, (x - DX0) * 256:(x - DX0 + 1) * 256] = \
            a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768

CH = gpd.read_file(os.path.join(BASEMAP, "ch_cantons.geojson")).to_crs(2056)
CH_UNION = CH.union_all()
LAKES = gpd.read_file(os.path.join(BASEMAP, "ch_lakes.geojson")).to_crs(2056).clip(CH_UNION)


# ------------------------------------------------------------------ helpers
def dem_grid(ext, res):
    xmin, xmax, ymin, ymax = ext
    X, Y = np.meshgrid(np.arange(xmin, xmax, res) + res / 2, np.arange(ymax, ymin, -res) - res / 2)
    lon, lat = TO_LL.transform(X, Y)
    n = 2 ** DZ * 256
    px = (lon + 180) / 360 * n - DX0 * 256
    py = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n - DY0 * 256
    return map_coordinates(DEM, [py - 0.5, px - 0.5], order=1, mode="nearest"), Y


def hillshade(el, res, az=315, alt=40, zf=1.3):
    dy, dx = np.gradient(el * zf, res)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, z = np.radians(az), np.radians(alt)
    return np.clip(np.sin(z) * np.sin(slope) + np.cos(z) * np.cos(slope) * np.cos(a - aspect), 0, 1)


def road_class(tags):
    h = tags.get("highway", "").replace("_link", "")
    if h in ("motorway", "trunk", "primary", "secondary", "tertiary"):
        return h
    return "minor" if h else None


def draw_base(ax, ext, res, styles, relief, river_lw, rail_lw):
    el, Y = dem_grid(ext, res)
    hs = hillshade(el, res)
    g = relief[0] + relief[1] * hs
    rgb = np.dstack([g, g, g])
    lake = (el < 436.5) & (Y > TO_UTM.transform(8.62, 46.89)[1])
    rgb[lake] = matplotlib.colors.to_rgb(WATER)
    ax.imshow(np.clip(rgb, 0, 1), extent=ext, origin="upper", zorder=0, interpolation="bilinear")
    ax.add_collection(PatchCollection([MplPolygon(w) for w in WATERS], facecolor=WATER, edgecolor="none", zorder=1))
    groups = {}
    for tags, xy in WAYS:
        if tags.get("waterway") == "river":
            groups.setdefault((WATER_LINE, river_lw, 2, "-"), []).append(xy)
        elif tags.get("railway") == "rail":
            if tags.get("tunnel") != "yes":
                groups.setdefault(("#8a8a8a", rail_lw, 3, (0, (3, 2))), []).append(xy)
        else:
            rc = road_class(tags)
            if rc not in styles:
                continue
            lw, c, z = styles[rc]
            if tags.get("tunnel") == "yes":
                groups.setdefault((c, lw * 0.6, z, (0, (1, 1.5))), []).append(xy)
            else:
                groups.setdefault((c, lw, z, "-"), []).append(xy)
    for (c, lw, z, ls), lines in groups.items():
        ax.add_collection(LineCollection(lines, colors=c, linewidths=lw, zorder=z, linestyles=ls,
                                         capstyle="round", joinstyle="round"))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#444444")


HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
HALO3 = [pe.withStroke(linewidth=3.0, foreground="white")]


def scalebar(ax, length, label, x_frac=0.06, y_frac=0.03):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x, y = x0 + x_frac * (x1 - x0), y0 + y_frac * (y1 - y0)
    h = (x1 - x0) * 0.022
    for k, fc in enumerate(("#222222", "white")):
        ax.add_patch(Rectangle((x + k * length / 2, y), length / 2, h, facecolor=fc, edgecolor="#222222",
                               lw=0.5, zorder=20))
    ax.text(x + length / 2, y + h * 1.5, label, ha="center", va="bottom", fontsize=6, zorder=20, path_effects=HALO)


def nice_length(width):
    for L in (2000, 1000, 500, 250):
        if L <= 0.4 * width:
            return L
    return 250


# ------------------------------------------------------------------ overview
def overview(ax):
    draw_base(ax, OV, P["ov_res"], OV_ROADS, P["ov_relief"], river_lw=0.7, rail_lw=0.35)
    for v, (l, r, b, t) in DOM.items():
        ax.add_patch(Rectangle((l, b), r - l, t - b, facecolor=ORANGE_FILL, edgecolor="none", alpha=0.6, zorder=1.5))
        ax.add_patch(Rectangle((l, b), r - l, t - b, facecolor="none", edgecolor=ORANGE_EDGE, lw=0.9, zorder=8))
        ax.text(r + 250, (b + t) / 2, LETTER[v], ha="left", va="center", fontsize=7.5, fontweight="bold",
                color=ORANGE_DARK, zorder=10, path_effects=HALO)
    for name, (lon, lat), side in [("Altdorf", (8.6440, 46.8805), "r"), ("Amsteg", (8.6703, 46.7709), "l")]:
        x, y = TO_UTM.transform(lon, lat)
        ax.plot(x, y, "o", ms=1.8, color="#333333", zorder=9)
        dx = 250 if side == "r" else -250
        ax.text(x + dx, y, name, ha="left" if side == "r" else "right", va="center", fontsize=6, style="italic",
                color="#333333", zorder=9, path_effects=HALO)
    px, py = TO_UTM.transform(8.5915, 46.6637)
    ax.plot(px, py, marker="v", ms=3.5, color="#222222", zorder=9)
    ax.text(px + 350, py - 150, "Gotthard\ntunnel portal", fontsize=5.5, style="italic", ha="left",
            va="top", color="#222222", zorder=9, path_effects=HALO)
    mw = np.vstack([xy for tags, xy in WAYS if tags.get("highway") == "motorway" and tags.get("tunnel") != "yes"])
    for lat in (46.745, 46.80):
        yt = TO_UTM.transform(8.63, lat)[1]
        x, y = mw[np.argmin(np.abs(mw[:, 1] - yt))]
        ax.text(x - 300, y, "A2", fontsize=5.5, fontweight="bold", color="white", ha="right", va="center", zorder=11,
                bbox=dict(boxstyle="round,pad=0.15", fc="#5c5c5c", ec="none"))
    scalebar(ax, 4000, "4 km", x_frac=0.6, y_frac=0.025)
    ax.annotate("N", xy=(0.92, 0.985), xytext=(0.92, 0.945), xycoords="axes fraction", ha="center", va="top",
                fontsize=7, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black",
                                                               mutation_scale=8), zorder=20)


def swiss_inset(ax):
    a = ax.inset_axes(P["swiss_rect"], zorder=40)
    CH.plot(ax=a, facecolor="#ececec", edgecolor="white", linewidth=0.3)
    gpd.GeoSeries([CH_UNION]).plot(ax=a, facecolor="none", edgecolor="#666666", linewidth=0.5)
    LAKES.plot(ax=a, facecolor=WATER, edgecolor="none")
    # footprint of the overview map
    cx = [OV[0], OV[1], OV[1], OV[0]]; cy = [OV[2], OV[2], OV[3], OV[3]]
    ex, ey = UTM_TO_LV95.transform(cx, cy)
    a.add_patch(MplPolygon(np.column_stack([ex, ey]), closed=True, facecolor=ORANGE_EDGE, edgecolor=ORANGE_DARK,
                           lw=0.6, alpha=0.9, zorder=5))
    a.set_xticks([]); a.set_yticks([])
    b = CH_UNION.bounds
    pad = 20000
    a.set_xlim(b[0] - pad, b[2] + pad); a.set_ylim(b[1] - pad, b[3] + pad)
    a.set_aspect("equal", adjustable="box", anchor="NW")
    for s in a.spines.values():
        s.set_linewidth(0.5); s.set_color("#444444")
    a.set_facecolor("white")


# ------------------------------------------------------------------ detail panels
def panel_ext(v):
    l, r, b, t = DOM[v]
    H = (t - b) / P["dom_frac"]
    W = H * P["panel_aspect"]
    cx, cy = (l + r) / 2, (b + t) / 2
    return (cx - W / 2, cx + W / 2, cy - H / 2, cy + H / 2)


def detail(ax, v):
    ext = panel_ext(v)
    res = (ext[3] - ext[2]) / P["in_px"]
    k = P["in_relief"]
    draw_base(ax, ext, res, IN_ROADS, (1 - k * 0.45, k * 0.45), river_lw=1.0, rail_lw=0.5)
    sel = ((B_FIRST[:, 0] > ext[0] - 50) & (B_FIRST[:, 0] < ext[1] + 50) &
           (B_FIRST[:, 1] > ext[2] - 50) & (B_FIRST[:, 1] < ext[3] + 50))
    ax.add_collection(PatchCollection([MplPolygon(BUILDINGS[i]) for i in np.where(sel)[0]],
                                      facecolor="#d2d2d2", edgecolor="none", zorder=2.5))
    l, r, b, t = DOM[v]
    ax.add_patch(Rectangle((l, b), r - l, t - b, facecolor=ORANGE_FILL, edgecolor="none", alpha=0.3, zorder=1.5))
    ax.add_collection(PatchCollection([MplPolygon(p) for p in HOUSES[v]], facecolor="#3f3f3f",
                                      edgecolor="#3f3f3f", linewidths=0.3, zorder=7))
    ax.add_patch(Rectangle((l, b), r - l, t - b, facecolor="none", edgecolor=ORANGE_EDGE, lw=1.0, zorder=8))
    t1 = ax.text(0.04, 0.975, LETTER[v], transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                 fontweight="bold", color=ORANGE_DARK, zorder=31, path_effects=HALO3)
    ax.annotate(f" {LABEL[v]}", xy=(1, 0), xycoords=t1, ha="left", va="bottom", fontsize=7.5, fontweight="bold",
                zorder=31, path_effects=HALO3)
    L = nice_length(ext[1] - ext[0])
    scalebar(ax, L, f"{L / 1000:g} km" if L >= 1000 else f"{L} m", x_frac=0.06, y_frac=0.02)


# ------------------------------------------------------------------ layout
def main():
    W = P["W_CM"] / 2.54
    g = P["gap_in"]
    ov_aspect = (OV[1] - OV[0]) / (OV[3] - OV[2])
    A = P["panel_aspect"]
    # W = ov_w + g + 3w + 2g ; ov_h = 2h + g ; h = w/A ; ov_w = ov_aspect*ov_h
    w = (W - 3 * g - ov_aspect * g) / (3 + 2 * ov_aspect / A)
    h = w / A
    ov_h = 2 * h + g
    ov_w = ov_aspect * ov_h
    fig = plt.figure(figsize=(W, ov_h))
    ax = fig.add_axes([0, 0, ov_w / W, 1])
    overview(ax)
    swiss_inset(ax)
    for i, v in enumerate(ORDER):
        row, col = divmod(i, 3)
        a = fig.add_axes([(ov_w + g + col * (w + g)) / W, (1 - row) * (h + g) / ov_h, w / W, h / ov_h])
        detail(a, v)
    save_fig(fig, "study_area", "study_area")
    plt.close(fig)


if __name__ == "__main__":
    main()
