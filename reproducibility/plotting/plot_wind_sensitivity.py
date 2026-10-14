"""
Wind sensitivity plot : Schattdorf CO at 9 AM, Scenario C, 20 July.
3 x 3 grid: rows = wind direction (compass bearing the wind blows FROM: 100, 210, 315 deg),
columns = wind speed (1, 10, 20 m/s). One colour scale for all 9 panels

Output:
  reproducibility/plotting/figures/wind_sensitivity/wind_schattdorf.pdf/png
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _shared import load_wind_snap, load_houses, draw_heatmap, save_fig, VILLAGES, LABEL

VILLAGE = "schattdorf"
DATE    = "07_20"
GAS     = "co"
TIME    = 32400                       # 9 AM
DIRECTIONS = [100, 210, 315]          # deg, compass bearing the wind blows from
SPEEDS     = [1, 10, 20]              # m/s
CO_BACKGROUND = 0.170                 # mg/m^3, config.CO_DEFAULT_MGM3
ZHAO = (315, 10)

data = {(d, s): load_wind_snap(VILLAGE, DATE, GAS, TIME, d, s) for d in DIRECTIONS for s in SPEEDS}
vmax = max(float(v.max()) for v in data.values())
houses = load_houses(VILLAGE)   # drawn as green outlines on every panel
cfg = VILLAGES[VILLAGE]

fig = plt.figure(figsize=(5.6, 7.5))
gs = gridspec.GridSpec(len(DIRECTIONS), len(SPEEDS) + 1, width_ratios=[1] * len(SPEEDS) + [0.06],
                       hspace=0.06, wspace=0.04)
cax = fig.add_subplot(gs[:, -1])

for r, d in enumerate(DIRECTIONS):
    for c, s in enumerate(SPEEDS):
        ax = fig.add_subplot(gs[r, c])
        draw_heatmap(ax, data[(d, s)], VILLAGE, vmax=vmax, vmin=CO_BACKGROUND,
                     house_df=houses, house_coloring=False)
        for patch in ax.patches:          # thin building outlines so the plume stays visible
            patch.set_linewidth(0.15)
        # arrow: direction the air moves towards (u_x = -sin(dir), u_y = -cos(dir))
        ux, uy = -math.sin(math.radians(d)), -math.cos(math.radians(d))
        x0 = cfg["GL"] + 0.14 * (cfg["GR"] - cfg["GL"])
        y0 = cfg["GB"] + 0.90 * (cfg["GT"] - cfg["GB"])
        L = 0.09 * (cfg["GR"] - cfg["GL"])
        ax.annotate("", xy=(x0 + ux * L, y0 + uy * L), xytext=(x0 - ux * L, y0 - uy * L),
                    arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.0))
        if r == 0:
            ax.set_title(f"{s} m s$^{{-1}}$", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"{d}°", fontsize=8, rotation=0, labelpad=14, va="center")
        if (d, s) == ZHAO:
            for sp in ax.spines.values():
                sp.set_edgecolor("tab:blue")
                sp.set_linewidth(1.8)

sm = ScalarMappable(cmap="hot_r", norm=Normalize(vmin=CO_BACKGROUND, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label(f"{LABEL[GAS]} [mg m$^{{-3}}$]", fontsize=7)
cbar.ax.tick_params(labelsize=6)

save_fig(fig, "wind_sensitivity", "wind_schattdorf")
plt.close()
print("Done.")
