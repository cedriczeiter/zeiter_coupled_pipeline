"""
Appendix -- representative 6 PM heatmaps, one figure per pollutant.
Each figure is a 2x3 grid of village panels (CO, NO2, PM2.5 separately),
Scenario C, 20 July 2024, exposure layer (0.3-1.2 m).
Scenario C: rt=True, rp=0.05, wd=False, date=07_20, t=64800.

Output:
  reproducibility/plotting/figures/appendix_heatmaps/appendix_heatmaps_{co,no2,pm25}.pdf/png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from _shared import VILLAGE_ORDER, VILLAGE_LABELS, load_snap, load_houses, draw_heatmap, save_fig

DATE  = "07_20"
TIME  = 64800
RT, RP, WD = True, 0.05, False

GASES = [
    ("co",  "co",   "CO"),
    ("nox", "no2",  "NO$_2$"),
    ("pmx", "pm25", "PM$_{2.5}$"),
]

BACKGROUND_MGM3 = {"co": 0.170, "nox": 0.0051, "pmx": 0.0055}


def main():
    houses_by_village = {village: load_houses(village) for village in VILLAGE_ORDER}

    for gas, slug, label in GASES:
        conc_by_village = {}
        for village in VILLAGE_ORDER:
            try:
                conc_by_village[village] = load_snap(village, DATE, gas, TIME, rt=RT, rp=RP, wd=WD)
            except FileNotFoundError as e:
                print(f"  WARN: {e}")
                conc_by_village[village] = None

        # Upper colour limit: 99.99th percentile, cells above saturate (better readability)
        pooled = np.concatenate([c.ravel() for c in conc_by_village.values() if c is not None])
        vmax = float(np.percentile(pooled, 99.99)) if pooled.size else BACKGROUND_MGM3[gas]

        fig = plt.figure(figsize=(6.8, 6.4))
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.08, hspace=0.15)

        im = None
        for i, village in enumerate(VILLAGE_ORDER):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            conc = conc_by_village[village]
            if conc is None:
                ax.text(0.5, 0.5, "data\nmissing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                im = draw_heatmap(ax, conc, village, vmax, vmin=BACKGROUND_MGM3[gas],
                                  house_df=houses_by_village[village])
            ax.set_title(VILLAGE_LABELS[village], fontsize=8)

        cax = fig.add_subplot(gs[:, 3])
        if im is not None:
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(f"{label} [mg m$^{{-3}}$]", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        save_fig(fig, "appendix_heatmaps", f"appendix_heatmaps_{slug}")
        plt.close(fig)
        print(f"  Done: {slug}")

    print("Appendix heatmaps complete.")


if __name__ == "__main__":
    main()
