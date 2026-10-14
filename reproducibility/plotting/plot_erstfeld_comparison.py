"""
Erstfeld CO: normal heatmap vs. filled-houses rendering.
Scenario C July (rt=True, rp=0.05, wd=False), t = 64800 s (6 PM).

Output:
  reproducibility/plotting/figures/comparison_plot/erstfeld_comparison.pdf/png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from _shared import load_snap, load_houses, draw_heatmap, save_fig

VILLAGE = "erstfeld"
DATE    = "07_20"
GAS     = "co"
TIME    = 64800

# Scenario C July
RT, RP, WD = True, 0.05, False

data   = load_snap(VILLAGE, DATE, GAS, TIME, rt=RT, rp=RP, wd=WD)
houses = load_houses(VILLAGE)

VMIN   = 0.170
vmax   = float(np.percentile(data, 99.5))

fig = plt.figure(figsize=(6.8, 4.5))
gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.04)

ax_left  = fig.add_subplot(gs[0])
ax_right = fig.add_subplot(gs[1])
cax      = fig.add_subplot(gs[2])

# Left panel: regular heatmap with house outlines
im = draw_heatmap(ax_left, data, VILLAGE, vmax, vmin=VMIN,
                  house_df=houses, house_coloring=False, only_houses=False)

# Right panel: filled-houses (houses coloured by local concentration)
draw_heatmap(ax_right, data, VILLAGE, vmax, vmin=VMIN,
             house_df=houses, house_coloring=True, only_houses=True)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("CO [mg m$^{-3}$]", fontsize=7)
cbar.ax.tick_params(labelsize=6)

save_fig(fig, "comparison_plot", "erstfeld_comparison")
plt.close()
print("Done.")
