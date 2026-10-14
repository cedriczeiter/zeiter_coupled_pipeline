"""Shared constants and helpers for all paper figure scripts."""

import ast
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

EXPOSURE_LAYER = 1   # z-layer index sampled for exposure and plots (0.3-1.2 m); keep equal to config.EXPOSURE_LAYER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPRO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # reproducibility/
ROOT_DIR  = os.path.normpath(os.path.join(REPRO_DIR, ".."))   # repository root
DATA_DIR  = os.path.join(REPRO_DIR, "data")
HOUSE_POLY_DIR = DATA_DIR
FIG_DIR   = os.path.join(REPRO_DIR, "plotting", "figures")
# real resident population (private, not part of the repository); only needed to rebuild data/residents_*.pkl
SYNPOP    = os.path.join(DATA_DIR, "synpop_clean_enriched.csv")

# ---------------------------------------------------------------------------
# Village registry
# ---------------------------------------------------------------------------
VILLAGES = {
    "erstfeld":   dict(GL=16850, GR=18630, GB=35000, GT=37720, NX=160, NY=250),
    "goeschenen": dict(GL=12600, GR=13700, GB=18200, GT=20000, NX=100, NY=180),
    "gurtnellen": dict(GL=15400, GR=16500, GB=24700, GT=27000, NX=110, NY=230),
    "schattdorf": dict(GL=16900, GR=19000, GB=38300, GT=41500, NX=210, NY=310),
    "silenen":    dict(GL=18900, GR=20100, GB=29900, GT=34300, NX=115, NY=430),
    "wassen":     dict(GL=13500, GR=15300, GB=20900, GT=24500, NX=175, NY=360),
}

VILLAGE_LABELS = {
    "erstfeld":   "Erstfeld",
    "goeschenen": "Göschenen",
    "gurtnellen": "Gurtnellen",
    "schattdorf": "Schattdorf",
    "silenen":    "Silenen",
    "wassen":     "Wassen",
}

VILLAGE_ORDER = ["schattdorf", "erstfeld", "silenen", "gurtnellen", "wassen", "goeschenen"]

VMAX  = {"co": 0.5, "nox": 0.07, "pmx": 0.010}
LABEL = {"co": "CO",  "nox": "NO$_2$", "pmx": "PM$_{2.5}$"}

# ---------------------------------------------------------------------------
# Matplotlib style defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ---------------------------------------------------------------------------
# Run-folder finder
# ---------------------------------------------------------------------------

def find_run_dir(village, date, rt, rp, wd):
    """
    Return the path of the matching output_* folder.

    Parameters match the output folder naming convention:
      rt  : bool / 0/1   random traffic flag
      rp  : float        rerouting share (0.0, 0.05, 0.1, 0.15)
      wd  : bool / 0/1   weekday flag
    """
    rt_str = str(bool(rt))   # "True" or "False"
    rp_str = str(float(rp))  # "0.05" etc.
    wd_str = str(bool(wd))

    pat = f"output_{village}_{date}_rt_{rt_str}_rp_{rp_str}_wd_{wd_str}_*"
    all_matches = [m for m in glob.glob(os.path.join(DATA_DIR, pat)) if os.path.isdir(os.path.join(m, "data"))]
    if not all_matches:
        raise FileNotFoundError(
            f"No run dir found for {village} {date} rt={rt} rp={rp} wd={wd}\n"
            f"  Pattern: {pat}\n"
            f"  Searched in: {DATA_DIR}"
        )

    # Prefer runs with more snapshots (complete > partial); break ties by path sort.
    def _snap_count(d):
        return len(glob.glob(os.path.join(d, "data", f"data_co_{EXPOSURE_LAYER}_*.csv")))
    return sorted(all_matches, key=lambda d: (_snap_count(d), d))[-1]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def find_wind_run_dir(village, date, direction, speed, rt=True, rp=0.05, wd=False):
    """Wind-sweep run folder output_wind_<v>_<date>_rt_..._w<dir>-<speed>_<stamp|curated>."""
    pat = (f"output_wind_{village}_{date}_rt_{bool(rt)}_rp_{float(rp)}_wd_{bool(wd)}"
           f"_w{direction:g}-{speed:g}_*")
    hits = sorted(m for m in glob.glob(os.path.join(DATA_DIR, pat)) if os.path.isdir(os.path.join(m, "data")))
    if not hits:
        raise FileNotFoundError(f"No wind run for {village} {date} {direction}:{speed}\n  Pattern: {pat}")
    return hits[-1]


def load_wind_snap(village, date, gas, time, direction, speed, **kw):
    """Exposure-layer snapshot (mg/m3) of a wind-sweep run."""
    path = os.path.join(find_wind_run_dir(village, date, direction, speed, **kw), "data",
                        f"data_{gas}_{EXPOSURE_LAYER}_{float(time)}.csv")
    return pd.read_csv(path).values


def load_snap(village, date, gas, time, rt=True, rp=0.05, wd=False):
    """
    Load the exposure-layer (0.3-1.2 m) concentration snapshot.

    Returns an (NX, NY) array already in mg/m³
    (save_all_data in plotting.py stores mg/m³, not mg/cell).
    """
    run_dir = find_run_dir(village, date, rt, rp, wd)
    fname   = f"data_{gas}_{EXPOSURE_LAYER}_{float(time)}.csv"
    path    = os.path.join(run_dir, "data", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot not found: {path}")
    # The CSV has a header row (column indices 0,1,...,NY-1)
    return pd.read_csv(path).values


def load_aqi(village, date, pollutant, rt=True, rp=0.05, wd=True):
    """Load per-cell max AQI grid from a run's final_results/ folder."""
    run_dir  = find_run_dir(village, date, rt, rp, wd)
    path     = os.path.join(run_dir, "data", "final_results",
                            f"{pollutant}_max_aqi.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AQI data not found: {path}\n"
            "Run the post-processing step first."
        )
    return pd.read_csv(path).values


def load_houses(village):
    """Load converted house-polygon CSV. Returns DataFrame or None."""
    cfg = VILLAGES[village]
    GL, GR, GB, GT = cfg["GL"], cfg["GR"], cfg["GB"], cfg["GT"]
    fname = f"converted_house_polygons_{GL}_{GR}_{GB}_{GT}.csv"
    path  = os.path.join(HOUSE_POLY_DIR, fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Core plotting primitive
# ---------------------------------------------------------------------------

def draw_heatmap(ax, data_mgm3, village, vmax, cmap="hot_r",
                 house_df=None, house_coloring=False, only_houses=False,
                 vmin=None):
    """
    Draw one concentration panel onto *ax*.

    Parameters
    ----------
    data_mgm3    : 2D array (NX, NY) in mg/m³
    village      : village key string
    vmax         : colour scale maximum
    vmin         : colour scale minimum (None = data min for imshow;
                   for house_coloring this sets the lower bound explicitly)
    house_df     : DataFrame from load_houses() or None
    house_coloring : colour each house by its grid-cell concentration
    only_houses  : make the background transparent (filled-houses mode)
    """
    cfg = VILLAGES[village]
    GL, GR, GB, GT = cfg["GL"], cfg["GR"], cfg["GB"], cfg["GT"]

    alpha = 0.0 if only_houses else 1.0
    im = ax.imshow(
        data_mgm3.T,
        cmap=cmap, origin="lower",
        extent=[GL, GR, GB, GT],
        vmin=vmin, vmax=vmax,
        alpha=alpha,
        aspect="equal",
        interpolation="nearest",
    )

    if house_df is not None:
        NX, NY = cfg["NX"], cfg["NY"]
        dx = (GR - GL) / NX
        dy = (GT - GB) / NY

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        # same lower bound as the imshow panel (autoscaled to the data minimum when vmin is None)
        _vmin = vmin if vmin is not None else float(np.nanmin(data_mgm3))
        norm = Normalize(vmin=_vmin, vmax=vmax)
        sm   = ScalarMappable(cmap=cmap, norm=norm)

        for _, row in house_df.iterrows():
            try:
                sumo_x = row.get("sumo_x")
                sumo_y = row.get("sumo_y")
                xi = int((sumo_x - GL) / dx)
                yi = int((sumo_y - GB) / dy)

                sumo_poly = row.get("sumo_polygon")
                sumo_bbox = row.get("sumo_bbox_polygon")
                coords = None
                if isinstance(sumo_poly, str) and len(sumo_poly) > 2:
                    try:
                        c = ast.literal_eval(sumo_poly)
                        if c:
                            coords = c
                    except Exception:
                        pass
                if coords is None and isinstance(sumo_bbox, str) and len(sumo_bbox) > 2:
                    try:
                        c = ast.literal_eval(sumo_bbox)
                        if c:
                            coords = c
                    except Exception:
                        pass
                if coords is None:
                    continue

                if house_coloring:
                    if 0 <= xi < NX and 0 <= yi < NY:
                        color = sm.to_rgba(data_mgm3[xi, yi])
                    else:
                        color = (0.5, 0.5, 0.5, 1.0)
                    patch = MplPolygon(coords, closed=True,
                                       facecolor=color, edgecolor="none", linewidth=0)
                else:
                    patch = MplPolygon(coords, closed=True,
                                       facecolor="none", edgecolor="green",
                                       linewidth=0.5, alpha=1.0)
                ax.add_patch(patch)
            except Exception:
                continue

    ax.set_xlim(GL, GR)
    ax.set_ylim(GB, GT)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.4")
    return im


def save_fig(fig, folder, name):
    """Save figure as PDF + PNG to reproducibility/plotting/figures/<folder>/."""
    out_dir = os.path.join(FIG_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    print(f"Saved → {folder}/{name}.pdf/png")
