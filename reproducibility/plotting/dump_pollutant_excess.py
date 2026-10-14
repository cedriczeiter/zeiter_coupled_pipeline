"""
Dump per-pollutant mean-excess-AQI-per-resident for every village/date pair.

This is the same computation the AQI bar charts use (compute_pollutant_metrics
in plot_aqi_bars.py: load_aqi() -> subtract I_BG[pollutant] -> mean over
residents), just printed as a table instead of plotted, plus a NO2-share-of-total-
excess column for the paper's "which pollutant" claims (Sec 5.4 / conclusion).

Run from reproducibility/plotting/:
    python dump_pollutant_excess.py
"""

from plot_aqi_bars import compute_pollutant_metrics, POLLUTANTS, VILLAGE_ORDER, I_BG, I_BG_TOTAL

DATES = ["07_20", "20_11"]

print(f"I_bg per pollutant: {I_BG}  (I_bg_total = {I_BG_TOTAL:.3f})")
print(f"{'village':<12}{'date':<8}{'co':>10}{'nox':>10}{'pmx':>10}{'total':>10}{'NO2 share':>12}")

shares = []
for village in VILLAGE_ORDER:
    for date in DATES:
        excesses = {}
        for pol in POLLUTANTS:
            m = compute_pollutant_metrics(village, date, pol)
            if m is None:
                print(f"{village:<12}{date:<8}  MISSING ({pol})")
                break
            excesses[pol] = m["mean_per_resident_excess"]
        else:
            total_excess = sum(excesses.values())
            nox_share = excesses["nox"] / total_excess * 100 if total_excess > 0 else float("nan")
            shares.append((village, date, nox_share))
            print(f"{village:<12}{date:<8}{excesses['co']:>10.3f}{excesses['nox']:>10.3f}"
                  f"{excesses['pmx']:>10.3f}{total_excess:>10.3f}{nox_share:>11.1f}%")

if shares:
    lo = min(shares, key=lambda s: s[2])
    hi = max(shares, key=lambda s: s[2])
    print(f"\nNO2 share range across all village/date pairs: {lo[2]:.1f}% - {hi[2]:.1f}%")
    print(f"  min at: {lo[0]} {lo[1]}")
    print(f"  max at: {hi[0]} {hi[1]}")
