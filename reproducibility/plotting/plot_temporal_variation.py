"""
Silenen temporal variation: CO and NO2 at 9 AM and 6 PM.
2x2 layout: rows = 9 AM | 6 PM,  columns = CO | NO2.
Scenario B2 (rt=True, rp=0.0, wd=True), date 07_20.

Output:
  reproducibility/plotting/figures/show_time_dependance/temporal_variation.pdf/png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from _shared import load_snap, load_houses, draw_heatmap, save_fig, LABEL

VILLAGE = "silenen"
DATE    = "07_20"
TIMES   = [32400, 64800]   # 9 AM, 6 PM
GASES   = ["co", "nox"]
TIME_LABELS = {32400: "9 AM", 64800: "6 PM"}

BACKGROUND = {"co": 0.170, "nox": 0.0051}

# Scenario B2: random traffic, weekday
RT, RP, WD = True, 0.0, True

houses = load_houses(VILLAGE)

data = {}
for gas in GASES:
    for t in TIMES:
        data[(gas, t)] = load_snap(VILLAGE, DATE, gas, t, rt=RT, rp=RP, wd=WD)

TEMPORAL_VMAX = {gas: float(max(np.percentile(data[(gas, t)], 99.5) for t in TIMES)) for gas in GASES}

fig = plt.figure(figsize=(7.2, 4.0))
gs  = gridspec.GridSpec(
    1, 7,
    width_ratios=[1, 1, 0.07, 0.45, 1, 1, 0.07],
    wspace=0.04,
)

# Panel order: CO 9AM, CO 6PM, NOx 9AM, NOx 6PM
panel_cols = [(("co",  32400), 0),
              (("co",  64800), 1),
              (("nox", 32400), 4),
              (("nox", 64800), 5)]

axes = {}
for (gas, t), col in panel_cols:
    ax = fig.add_subplot(gs[0, col])
    axes[(gas, t)] = ax

cax_co  = fig.add_subplot(gs[0, 2])
cax_nox = fig.add_subplot(gs[0, 6])

for gas in GASES:
    vmax = TEMPORAL_VMAX[gas]
    for t in TIMES:
        ax = axes[(gas, t)]
        im = draw_heatmap(ax, data[(gas, t)], VILLAGE, vmax, vmin=BACKGROUND[gas],
                          house_df=houses, house_coloring=False, only_houses=False)
        axes[(gas, t)]._last_im = im

# Shared colorbars
for gas, cax in [("co", cax_co), ("nox", cax_nox)]:
    im_ref = axes[(gas, TIMES[0])]._last_im
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label(f"{LABEL[gas]} [mg m$^{{-3}}$]", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

# Time labels below each panel
for (gas, t), col in panel_cols:
    ax = axes[(gas, t)]
    ax.set_xlabel(TIME_LABELS[t], fontsize=8, labelpad=3)

# Gas headers above left panel of each pair
axes[("co",  TIMES[0])].set_title(LABEL["co"],  fontsize=9, pad=3)
axes[("nox", TIMES[0])].set_title(LABEL["nox"], fontsize=9, pad=3)

save_fig(fig, "show_time_dependance", "temporal_variation")
plt.close()
print("Done.")
