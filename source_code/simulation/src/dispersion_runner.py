"""
Glue between the per-vehicle emissions and the dispersion solver: one Pollutant
record per species and a single function that advances all species one step.
Also records / replays per-step emissions so dispersion parameters can be
explored without re-running SUMO.
"""

from dataclasses import dataclass

import numpy as np

from emission_models import process_gas_step, Z_THICKNESSES

EMISSION_KEYS = ("co", "nox", "pmx")


@dataclass(frozen=True)
class Pollutant:
    key: str            # "co" | "nox" | "pmx"; file names use this key
    background: float   # mg/m^3
    Dx: float           # m^2/s
    Dy: float           # m^2/s
    lam: float          # 1/s, loss of the excess over background


def pollutants_from_config(cfg):
    """The three species with their config.py background, D_x, D_y and lambda."""
    return [
        Pollutant("co",  cfg.CO_DEFAULT_MGM3,  cfg.DX_CO,  cfg.DY_CO,  cfg.LAMBDA_CO),
        Pollutant("nox", cfg.NO2_DEFAULT_MGM3, cfg.DX_NOX, cfg.DY_NOX, cfg.LAMBDA_NOX),
        Pollutant("pmx", cfg.PM2_DEFAULT_MGM3, cfg.DX_PMX, cfg.DY_PMX, cfg.LAMBDA_PMX),
    ]


def init_field(background, nx, ny, nz, cell_len_x_m, cell_len_y_m):
    """Grid at the background concentration, in mg/cell, shape (nx, ny, nz)."""
    vol = cell_len_x_m * cell_len_y_m * Z_THICKNESSES[:nz]
    return np.broadcast_to(background * vol, (nx, ny, nz)).copy()


def init_fields(pollutants, nx, ny, nz, cell_len_x_m, cell_len_y_m):
    return {p.key: init_field(p.background, nx, ny, nz, cell_len_x_m, cell_len_y_m)
            for p in pollutants}


def advance_fields(fields, pollutants, x_vec, y_vec, emissions, cell_len_x_m, cell_len_y_m,
                   dt=1.0, wind=(0.0, 0.0)):
    """One dispersion step for every species; `emissions` maps key -> per-vehicle mg."""
    return {
        p.key: process_gas_step(
            fields[p.key], x_vec, y_vec, emissions[p.key],
            cell_len_x_m, cell_len_y_m, p.background,
            p.Dx, p.Dy, None, dt, lam=p.lam, wind=wind)
        for p in pollutants
    }


class EmissionRecorder:
    """Collects (x, y, co, nox, pmx) of all vehicles per step; saves to one .npz."""

    def __init__(self):
        self._times, self._counts = [], []
        self._cols = {k: [] for k in ("x", "y") + EMISSION_KEYS}

    def add(self, sim_time, x_vec, y_vec, emissions):
        self._times.append(sim_time)
        self._counts.append(len(x_vec))
        self._cols["x"].append(np.asarray(x_vec, dtype=np.float32))
        self._cols["y"].append(np.asarray(y_vec, dtype=np.float32))
        for k in EMISSION_KEYS:
            self._cols[k].append(np.asarray(emissions[k], dtype=np.float32))

    def save(self, path):
        cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0, np.float32)
        np.savez_compressed(path, times=np.array(self._times), counts=np.array(self._counts),
                            **{k: cat(v) for k, v in self._cols.items()})


def iter_recorded_steps(path):
    """Yield (sim_time, x_vec, y_vec, emissions) per recorded step."""
    with np.load(path) as f:          # NpzFile decompresses on every access: load once
        d = {k: f[k] for k in f.files}
    offsets = np.concatenate([[0], np.cumsum(d["counts"])])
    for i, t in enumerate(d["times"]):
        s = slice(offsets[i], offsets[i + 1])
        yield float(t), d["x"][s], d["y"][s], {k: d[k][s] for k in EMISSION_KEYS}
