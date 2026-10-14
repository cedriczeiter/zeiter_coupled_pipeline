"""
Which simulation-run configurations and (gas, time) snapshot pairs the
reproducibility package needs, plus the logic that copies a finished run's
output into reproducibility/data/.

Used by:
  - run_sim.py: calls sync_run_to_reproducibility() right after
    post_processing_wrapper(), so every run automatically updates
    reproducibility/data/ if it matches (or could matter for) a figure.
    Wind-sweep runs are synced by sync_wind_run_to_reproducibility().
  - `python curated_manifest.py`: syncs the newest finished run of every
    configuration found in source_code/simulation/output_*/.

If you change what a figure plots (a new rerouting share, a different
snapshot time, ...), update build_manifest() here -- that is the one place
the auto-sync hook reads from.
"""
import glob
import os
import re
import shutil

from config import EXPOSURE_LAYER

VILLAGE_ORDER = ["schattdorf", "erstfeld", "silenen", "gurtnellen", "wassen", "goeschenen"]
DATES = ["07_20", "20_11"]
DATE_WD = {"07_20": False, "20_11": True}


def build_manifest():
    """Returns {(village, date, rt, rp, wd): set of (gas, time)} needed by the paper."""
    needs = {}

    def need(v, date, rt, rp, wd, files):
        needs.setdefault((v, date, bool(rt), float(rp), bool(wd)), set()).update(files)

    # Scenario C (rt=True, rp=0.05) for all villages x both dates -> final_results
    # (cross-village AQI / dump_pollutant_excess), plus appendix heatmap
    # snapshots (07_20 only, co/nox/pmx @ 64800), plus the Erstfeld comparison (co @ 64800).
    for v in VILLAGE_ORDER:
        for date in DATES:
            wd = DATE_WD[date]
            files = set()
            if date == "07_20":
                files |= {("co", 64800), ("nox", 64800), ("pmx", 64800)}
            need(v, date, True, 0.05, wd, files)

    # Temporal variation: Silenen 07_20, Scenario B2 (rt=True, rp=0.0, wd=True), co/nox @ 9am/6pm
    need("silenen", "07_20", True, 0.0, True, {("co", 32400), ("co", 64800),
                                                ("nox", 32400), ("nox", 64800)})

    # Random traffic: Wassen 07_20, three scenarios, nox @ 6pm
    need("wassen", "07_20", False, 0.0, False, {("nox", 64800)})
    need("wassen", "07_20", True, 0.0, False, {("nox", 64800)})
    need("wassen", "07_20", True, 0.0, True, {("nox", 64800)})

    # Rerouting (sweep, 0/5/10/15 %): plotted figure (nox) at t=72000 (8 pm)
    for rp in [0.0, 0.05, 0.10, 0.15]:
        need("schattdorf", "07_20", True, rp, False, {("nox", 72000)})

    return needs


WIND_DIRECTIONS = [100, 210, 315]   # compass bearing the wind blows from [deg]
WIND_SPEEDS = [1, 10, 20]           # [m/s]
WIND_SNAPSHOTS = {("co", 32400)}    # 9 AM CO, as plotted in the wind-sensitivity figure


def build_wind_manifest():
    """Wind-sweep runs plotted in the paper (Schattdorf, Scenario C, 20 July):
    [(direction, speed, {(gas, time)})]."""
    return [(d, s, set(WIND_SNAPSHOTS)) for d in WIND_DIRECTIONS for s in WIND_SPEEDS]


def _repro_data_dir():
    """reproducibility/data/, three levels up from this file (simulation/src/../../.. )."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(src_dir, "..", "..", ".."))
    return os.path.join(root, "reproducibility", "data")


def sync_run_to_reproducibility(village, date, rt, rp, wd, output_path,
                                 house_poly_dir=None, verbose=True):
    """
    Best-effort copy of a just-finished run into reproducibility/data/.

    Copies final_results/*.csv only for Scenario C runs (rt=True, rp=0.05, the
    weekday flag matching the date), which the cross-village AQI plot and the appendix bar plots
    read, plus the (gas, time) snapshot CSVs build_manifest() lists for this
    exact (village, date, rt, rp, wd). Runs outside the manifest are ignored,
    so reproducibility/data/ only ever holds files a plot script reads.

    Never raises: reproducibility/data/ syncing is a convenience, not part
    of the simulation itself, so a missing folder or an I/O hiccup here
    must not fail a multi-hour simulation run.
    """
    try:
        repro_data = _repro_data_dir()
        if not os.path.isdir(os.path.dirname(repro_data)):  # no reproducibility/ folder at all
            return

        key = (village, date, bool(rt), float(rp), bool(wd))
        manifest = build_manifest()
        needed_files = manifest.get(key, set())

        if key not in manifest:
            if verbose:
                print(f"[reproducibility] {village} {date} rt={rt} rp={rp} wd={wd} not in manifest, skipped")
            return

        is_scenario_c = bool(rt) and float(rp) == 0.05 and bool(wd) == DATE_WD.get(date)
        src_fr = os.path.join(output_path, "data", "final_results")

        rt_str, rp_str, wd_str = str(bool(rt)), str(float(rp)), str(bool(wd))
        dest_dir = os.path.join(
            repro_data, f"output_{village}_{date}_rt_{rt_str}_rp_{rp_str}_wd_{wd_str}_curated", "data"
        )
        os.makedirs(dest_dir, exist_ok=True)

        n_fr = 0
        if is_scenario_c and os.path.isdir(src_fr):
            dest_fr = os.path.join(dest_dir, "final_results")
            os.makedirs(dest_fr, exist_ok=True)
            for fn in os.listdir(src_fr):
                if fn.endswith(".csv"):
                    shutil.copy2(os.path.join(src_fr, fn), os.path.join(dest_fr, fn))
                    n_fr += 1

        copied = []
        for gas, t in sorted(needed_files):
            fname = f"data_{gas}_{EXPOSURE_LAYER}_{float(t)}.csv"
            src_f = os.path.join(output_path, "data", fname)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(dest_dir, fname))
                copied.append(fname)

        # House-polygon CSV for this village (small, shared across all its runs)
        if house_poly_dir and os.path.isdir(house_poly_dir):
            for fn in os.listdir(house_poly_dir):
                if fn.startswith("converted_house_polygons_") and fn.endswith(".csv"):
                    shutil.copy2(os.path.join(house_poly_dir, fn), os.path.join(repro_data, fn))

        if verbose:
            tag = f"{village} {date} rt={rt} rp={rp} wd={wd}"
            print(f"[reproducibility] synced {tag}: "
                  f"{n_fr} final_results file(s), {len(copied)} snapshot(s) -> {dest_dir}")
    except Exception as e:
        print(f"[reproducibility] sync skipped due to error: {e}")


def sync_wind_run_to_reproducibility(village, date, rt, rp, wd, direction, speed, output_path, verbose=True):
    """
    Best-effort copy of a finished wind-sweep run's plotted snapshot(s) into
    reproducibility/data/output_wind_..._w<dir>-<speed>_curated/data/.
    Only runs listed in build_wind_manifest() (Schattdorf, Scenario C, 20 July)
    are copied. Never raises, for the same reason as sync_run_to_reproducibility().
    """
    try:
        if (village, date, bool(rt), float(rp), bool(wd)) != ("schattdorf", "07_20", True, 0.05, False):
            return
        files = {(d, s): f for d, s, f in build_wind_manifest()}.get((direction, speed))
        if not files:
            return
        repro_data = _repro_data_dir()
        if not os.path.isdir(os.path.dirname(repro_data)):
            return
        dest_dir = os.path.join(
            repro_data,
            f"output_wind_{village}_{date}_rt_{bool(rt)}_rp_{float(rp)}_wd_{bool(wd)}"
            f"_w{direction:g}-{speed:g}_curated", "data")
        os.makedirs(dest_dir, exist_ok=True)
        copied = 0
        for gas, t in sorted(files):
            fname = f"data_{gas}_{EXPOSURE_LAYER}_{float(t)}.csv"
            src_f = os.path.join(output_path, "data", fname)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(dest_dir, fname))
                copied += 1
        if verbose:
            print(f"[reproducibility] synced wind {direction:g}:{speed:g} -> {dest_dir} ({copied} snapshot(s))")
    except Exception as e:
        print(f"[reproducibility] wind sync skipped due to error: {e}")


def sync_all_existing_runs(sim_dir=None):
    """Sync the newest run of every configuration under source_code/simulation/output_*/."""
    sim_dir = sim_dir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    pat = re.compile(r"output_(?P<wind>wind_)?(?P<v>[a-z]+)_(?P<date>\d\d_\d\d)_rt_(?P<rt>True|False)"
                     r"_rp_(?P<rp>[\d.]+)_wd_(?P<wd>True|False)(?:_w(?P<d>[\d.]+)-(?P<s>[\d.]+))?_\d{4}_.*$")
    newest = {}
    for path in sorted(glob.glob(os.path.join(sim_dir, "output_*"))):   # stamp in name -> sorted = oldest first
        m = pat.match(os.path.basename(path))
        if m and os.path.isdir(os.path.join(path, "data")):
            newest[tuple(m.group(k) for k in ("wind", "v", "date", "rt", "rp", "wd", "d", "s"))] = (path, m)
    for path, m in newest.values():
        args = dict(village=m["v"], date=m["date"], rt=m["rt"] == "True", rp=float(m["rp"]),
                    wd=m["wd"] == "True", output_path=path)
        if m["wind"]:
            sync_wind_run_to_reproducibility(direction=float(m["d"]), speed=float(m["s"]), **args)
        else:
            sync_run_to_reproducibility(**args)


if __name__ == "__main__":
    sync_all_existing_runs()
