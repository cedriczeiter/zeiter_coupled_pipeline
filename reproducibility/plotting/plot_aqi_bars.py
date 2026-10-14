"""
Cross-village AQI exposure comparison (Scenario C, both dates).

Produces one figure with 3 panels (total sum | mean per resident |
mean excess per resident), showing both dates side-by-side per village,
and per-pollutant appendix bar charts (CO, NO2, PM2.5).

Input (reproducibility/data/):
  output_<village>_<date>_rt_True_rp_0.05_wd_<...>_curated/data/final_results/{co,nox,pmx}_max_aqi.csv
  residents_<village>.pkl  (grid cell of every resident; rebuilt from the private SynPop file if missing)

Output:
  reproducibility/plotting/figures/bar_plots/combined_aqi.pdf/png
  reproducibility/plotting/figures/bar_plots/appendix_aqi_co.pdf/png
  reproducibility/plotting/figures/bar_plots/appendix_aqi_nox.pdf/png
  reproducibility/plotting/figures/bar_plots/appendix_aqi_pmx.pdf/png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _shared import (load_aqi, save_fig, VILLAGES, VILLAGE_LABELS, VILLAGE_ORDER,
                     SYNPOP, ROOT_DIR, DATA_DIR)
from aqi_lite import calculate_aqi_for_pollutant_US_AQI, mg_per_m3_to_ppm

POLLUTANTS  = ["co", "nox", "pmx"]

_BACKGROUND_MGM3 = {"co": 0.170, "nox": 0.0051, "pmx": 0.0055}
_AQI_CONVERSION = {
    "co":  mg_per_m3_to_ppm(28.01),        # mg/m3 -> ppm
    "nox": mg_per_m3_to_ppm(46.01) * 1000, # mg/m3 -> ppb
    "pmx": 1e3,                            # mg/m3 -> ug/m3
}
_AQI_POLLUTANT_NAME = {"co": "CO", "nox": "NO2", "pmx": "PM2.5"}

I_BG = {
    pol: calculate_aqi_for_pollutant_US_AQI(
        _BACKGROUND_MGM3[pol] * _AQI_CONVERSION[pol], _AQI_POLLUTANT_NAME[pol]
    )
    for pol in POLLUTANTS
}
I_BG_TOTAL = sum(I_BG.values())
POL_LABELS  = {"co": "CO", "nox": "NO$_2$", "pmx": "PM$_{2.5}$"}
DATES       = ["07_20", "20_11"]
DATE_LABELS = {"07_20": "20 July (busy)", "20_11": "20 Nov (quiet)"}
DATE_COLORS = {"07_20": "#c0392b", "20_11": "#2980b9"}

# July 20 is a Saturday (wd=False), November 20 is a Wednesday (wd=True)
DATE_WD = {"07_20": False, "20_11": True}


# ---------------------------------------------------------------------------
# Population loading with caching
# ---------------------------------------------------------------------------

def _load_population_for_village(village):
    """
    Return list of (grid_x, grid_y) for every resident in the village bounds.
    Uses a pickle cache per village to avoid repeated lat/lon conversions.
    """
    cfg   = VILLAGES[village]
    GL, GR, GB, GT = cfg["GL"], cfg["GR"], cfg["GB"], cfg["GT"]
    NX, NY         = cfg["NX"], cfg["NY"]

    cache = os.path.join(DATA_DIR, f"residents_{village}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"  Converting SynPop coordinates for {village} …")

    from sumolib import net as sumonet
    village_cap = village.capitalize()
    netfile = os.path.join(ROOT_DIR, "source_code", "simulation", "villages",
                           village_cap, f"{village}_osm.net.xml")
    if not os.path.exists(netfile):
        raise FileNotFoundError(f"Network file not found: {netfile}")
    network = sumonet.readNet(netfile)

    synpop = pd.read_csv(SYNPOP)   # has columns: lat, lon
    dx = (GR - GL) / NX
    dy = (GT - GB) / NY

    residents = []
    for _, row in synpop.iterrows():
        x, y = network.convertLonLat2XY(row["lon"], row["lat"])
        if GL <= x <= GR and GB <= y <= GT:
            gx = min(int((x - GL) / dx), NX - 1)
            gy = min(int((y - GB) / dy), NY - 1)
            residents.append((gx, gy))

    with open(cache, "wb") as f:
        pickle.dump(residents, f)
    print(f"    → {len(residents)} residents")
    return residents


# ---------------------------------------------------------------------------
# AQI aggregation for one run
# ---------------------------------------------------------------------------

def compute_village_metrics(village, date):
    """
    Returns dict with keys:
      total_sum, mean_per_resident, mean_per_resident_excess
    for the summed-sub-index AQI (CO + NOx + PMx).
    """
    grids = {}
    wd = DATE_WD.get(date, True)
    for pol in POLLUTANTS:
        try:
            grids[pol] = load_aqi(village, date, pol, wd=wd)
        except FileNotFoundError:
            return None

    # Summed AQI grid
    total_grid = sum(grids[p] for p in POLLUTANTS)

    residents = _load_population_for_village(village)
    if not residents:
        return None

    aqi_values = np.array([total_grid[gx, gy] for gx, gy in residents])
    n_total    = len(aqi_values)

    return {
        "total_sum":                float(aqi_values.sum()),
        "mean_per_resident":        float(aqi_values.mean()) if n_total > 0 else 0.0,
        "mean_per_resident_excess": float((aqi_values - I_BG_TOTAL).mean())
                                    if n_total > 0 else 0.0,
        "n_residents":              n_total,
    }


def compute_pollutant_metrics(village, date, pollutant):
    """Like compute_village_metrics but for a single pollutant's AQI grid."""
    wd = DATE_WD.get(date, True)
    try:
        grid = load_aqi(village, date, pollutant, wd=wd)
    except FileNotFoundError:
        return None
    residents  = _load_population_for_village(village)
    if not residents:
        return None
    aqi_values = np.array([grid[gx, gy] for gx, gy in residents])
    return {
        "total_sum":                float(aqi_values.sum()),
        "mean_per_resident":        float(aqi_values.mean()),
        "mean_per_resident_excess": float((aqi_values - I_BG[pollutant]).mean()),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

METRICS = [
    ("total_sum",                "Total AQI (sum over all residents)"),
    ("mean_per_resident",        "Mean AQI per resident"),
    ("mean_per_resident_excess", "Mean excess AQI per resident"),
]

def _make_bar_chart(ax, data_by_date, metric_key, ylabel, village_order):
    """
    Draw grouped bars (one group per village, one bar per date).
    data_by_date: dict  date → {village: value}
    """
    x       = np.arange(len(village_order))
    n_dates = len(DATES)
    width   = 0.35
    offsets = np.linspace(-(n_dates - 1) / 2, (n_dates - 1) / 2, n_dates) * width

    for i, date in enumerate(DATES):
        vals = [data_by_date[date].get(v, 0.0) for v in village_order]
        ax.bar(x + offsets[i], vals, width=width,
               label=DATE_LABELS[date], color=DATE_COLORS[date],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [VILLAGE_LABELS[v] for v in village_order],
        rotation=30, ha="right", fontsize=7,
    )
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading AQI metrics …")
    # Shape: results[date][village] = metrics_dict
    results = {date: {} for date in DATES}
    for date in DATES:
        for village in VILLAGE_ORDER:
            m = compute_village_metrics(village, date)
            if m is not None:
                results[date][village] = m

    # Per-pollutant results
    pol_results = {pol: {date: {} for date in DATES} for pol in POLLUTANTS}
    for pol in POLLUTANTS:
        for date in DATES:
            for village in VILLAGE_ORDER:
                m = compute_pollutant_metrics(village, date, pol)
                if m is not None:
                    pol_results[pol][date][village] = m

    print(f"I_bg per pollutant: {I_BG}  (total I_bg = {I_BG_TOTAL:.3f})")
    print(f"{'village':<12}{'date':<8}{'total_sum':>12}{'mean/res':>12}{'mean_excess':>14}")
    for date in DATES:
        for village in VILLAGE_ORDER:
            m = results[date].get(village)
            if m is None:
                continue
            print(f"{village:<12}{date:<8}{m['total_sum']:>12.0f}"
                  f"{m['mean_per_resident']:>12.2f}{m['mean_per_resident_excess']:>14.2f}")

    # ---------------------------------------------------------------------------
    # Combined figure (3 metric panels)
    # ---------------------------------------------------------------------------
    print("Generating combined figure …")
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), sharey=False)
    fig.subplots_adjust(wspace=0.38, bottom=0.22)

    metric_ylabels = {
        "total_sum":                "Summed AQI",
        "mean_per_resident":        "Mean AQI / resident",
        "mean_per_resident_excess": "Mean excess AQI / resident",
    }

    for ax, (mk, _) in zip(axes, METRICS):
        data_plot = {
            date: {v: results[date][v][mk] for v in VILLAGE_ORDER if v in results[date]}
            for date in DATES
        }
        _make_bar_chart(ax, data_plot, mk, metric_ylabels[mk], VILLAGE_ORDER)

    # Legend on first panel only
    axes[0].legend(fontsize=6, frameon=False)

    save_fig(fig, "bar_plots", "combined_aqi")
    plt.close()

    # ---------------------------------------------------------------------------
    # Appendix : per-pollutant bar charts
    # ---------------------------------------------------------------------------
    for pol in POLLUTANTS:
        print(f"Generating appendix bar chart: {pol} …")
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), sharey=False)
        fig.subplots_adjust(wspace=0.38, bottom=0.22)

        for ax, (mk, _) in zip(axes, METRICS):
            data_plot = {
                date: {v: pol_results[pol][date][v][mk]
                       for v in VILLAGE_ORDER if v in pol_results[pol][date]}
                for date in DATES
            }
            _make_bar_chart(ax, data_plot, mk, metric_ylabels[mk], VILLAGE_ORDER)

        axes[0].set_title(POL_LABELS[pol], fontsize=8)
        axes[0].legend(fontsize=6, frameon=False)

        save_fig(fig, "bar_plots", f"appendix_aqi_{pol}")
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
