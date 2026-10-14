"""
Wassen NO2: effect of randomised local traffic at t=64800 (6 PM).
Houses-only rendering, zoomed to settlement extent.
1x3 layout: no random | weekend random | weekday random.

Scenarios:
  A  : rt=False, rp=0.0, wd=False  (no random traffic)
  B1 : rt=True,  rp=0.0, wd=False  (random, weekend profile)
  B2 : rt=True,  rp=0.0, wd=True   (random, weekday profile)

Output:
  reproducibility/plotting/figures/show_weekend_effect/random_traffic.pdf/png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _shared import load_snap, load_houses, draw_heatmap, save_fig, VILLAGES

VILLAGE = "wassen"
DATE    = "07_20"
GAS     = "nox"
TIME    = 64800   # 6 PM : largest mean diff at house level

SCENARIOS = [
    (False, 0.0, False, "No random traffic"),
    (True,  0.0, False, "Weekend random traffic"),
    (True,  0.0, True,  "Weekday random traffic"),
]

houses = load_houses(VILLAGE)

# Load all scenario data
all_data = [load_snap(VILLAGE, DATE, GAS, TIME, rt=rt, rp=rp, wd=wd)
            for rt, rp, wd, _ in SCENARIOS]

# Colour scale calibrated to house-level concentrations (not road peaks)
cfg = VILLAGES[VILLAGE]
h2 = houses.dropna(subset=["sumo_x", "sumo_y"])
dx = (cfg["GR"] - cfg["GL"]) / cfg["NX"]
dy = (cfg["GT"] - cfg["GB"]) / cfg["NY"]
xi = ((h2.sumo_x - cfg["GL"]) / dx).astype(int).clip(0, cfg["NX"]-1).values
yi = ((h2.sumo_y - cfg["GB"]) / dy).astype(int).clip(0, cfg["NY"]-1).values

house_vals = [d[xi, yi] for d in all_data]
VMIN_V = min(v.min() for v in house_vals)   # background level
VMAX_V = max(v.max() for v in house_vals) * 1.05

# Zoom to the dense village core only.
ZOOM_X0, ZOOM_X1 = 13860, 14270
ZOOM_Y0, ZOOM_Y1 = 23000, 23420

fig = plt.figure(figsize=(6.8, 2.6))
gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.04)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
cax  = fig.add_subplot(gs[3])

for ax, data_conc, (rt, rp, wd, sublabel) in zip(axes, all_data, SCENARIOS):
    draw_heatmap(ax, data_conc, VILLAGE, vmax=VMAX_V, vmin=VMIN_V,
                 house_df=houses, house_coloring=True, only_houses=True)
    ax.set_xlim(ZOOM_X0, ZOOM_X1)
    ax.set_ylim(ZOOM_Y0, ZOOM_Y1)
    ax.set_aspect("auto")   # allow panel to stretch; background is transparent
    ax.text(0.5, -0.02, sublabel, transform=ax.transAxes,
            ha="center", va="top", fontsize=7)

sm = ScalarMappable(cmap="hot_r", norm=Normalize(vmin=VMIN_V, vmax=VMAX_V))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("NO$_2$ [mg m$^{-3}$]", fontsize=7)
cbar.ax.tick_params(labelsize=6)

save_fig(fig, "show_weekend_effect", "random_traffic")
plt.close()
print("Done.")
