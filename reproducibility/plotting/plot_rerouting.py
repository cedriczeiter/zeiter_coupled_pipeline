"""
Schattdorf NO2: dynamic rerouting levels at t=72000 (8 PM).
Houses-only rendering.
2x2 layout: 0% | 5% (top row),  10% | 15% (bottom row).
All scenarios: rt=True, wd=False, date=07_20.

Output:
  reproducibility/plotting/figures/show_rerouting/rerouting.pdf/png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _shared import load_snap, load_houses, draw_heatmap, save_fig, VILLAGES

VILLAGE = "schattdorf"
DATE    = "07_20"
GAS     = "nox"
TIME    = 72000

PANELS = [
    (0.0,  "0 % rerouted"),
    (0.05, "5 % rerouted"),
    (0.10, "10 % rerouted"),
    (0.15, "15 % rerouted"),
]

houses = load_houses(VILLAGE)
all_data = [load_snap(VILLAGE, DATE, GAS, TIME, rt=True, rp=rp, wd=False)
            for rp, _ in PANELS]

# Colour scale calibrated to house-level concentrations
cfg = VILLAGES[VILLAGE]
h2 = houses.dropna(subset=["sumo_x", "sumo_y"])
dx = (cfg["GR"] - cfg["GL"]) / cfg["NX"]
dy = (cfg["GT"] - cfg["GB"]) / cfg["NY"]
xi = ((h2.sumo_x - cfg["GL"]) / dx).astype(int).clip(0, cfg["NX"]-1).values
yi = ((h2.sumo_y - cfg["GB"]) / dy).astype(int).clip(0, cfg["NY"]-1).values

house_vals = [d[xi, yi] for d in all_data]
VMIN_V = min(v.min() for v in house_vals)
VMAX_V = max(v.max() for v in house_vals) * 0.75

# Zoom to main settlement (p2:p98 of house coords, cutting empty southern fringe)
ZOOM_X0, ZOOM_X1 = 16900, 18830
ZOOM_Y0, ZOOM_Y1 = 39450, 41550

fig = plt.figure(figsize=(6.2, 6.0))
gs  = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.06], hspace=0.12, wspace=0.04)
panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
cax = fig.add_subplot(gs[:, 2])

for (row, col), data_conc, (rp, sublabel) in zip(panel_positions, all_data, PANELS):
    ax = fig.add_subplot(gs[row, col])
    draw_heatmap(ax, data_conc, VILLAGE, vmax=VMAX_V, vmin=VMIN_V,
                 house_df=houses, house_coloring=True, only_houses=True)
    ax.set_xlim(ZOOM_X0, ZOOM_X1)
    ax.set_ylim(ZOOM_Y0, ZOOM_Y1)
    ax.set_aspect("auto")
    ax.text(0.5, 0.04, sublabel, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.25",
                      linewidth=0.6, alpha=0.85))

sm = ScalarMappable(cmap="hot_r", norm=Normalize(vmin=VMIN_V, vmax=VMAX_V))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("NO$_2$ [mg m$^{-3}$]", fontsize=7)
cbar.ax.tick_params(labelsize=6)

save_fig(fig, "show_rerouting", "rerouting")
plt.close()
print("Done.")
