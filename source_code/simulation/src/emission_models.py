"""
Pollutant transport using an implicit ADI (Lie-splitting) scheme on a
geometric vertical grid: horizontal diffusion (+ optional uniform upwind
wind), height-dependent vertical diffusion K_z(z) (Monin-Obukhov, see
atmosphere.py) and an optional first-order loss.

Vertical layers (boundaries in metres):  0 → 0.3 → 1.2 → 4.8 → 19.2 → 76.8 → 200
Boundary conditions:
  - Lateral (x, y)  : Dirichlet at background concentration
  - Bottom  (z = 0) : Neumann  (no-flux, reflecting)
  - Top     (z=200) : Dirichlet at background concentration

The scheme is unconditionally stable; dt is capped at 1 s so that the
microsimulation source injection stays temporally realistic.
"""

import numpy as np
from scipy.linalg import solve_banded

from atmosphere import kz_profile

import config as _cfg


def _grid():
    """Grid bounds, read through the module so per-village overrides of config are visible."""
    return _cfg.GRID_LEFT, _cfg.GRID_RIGHT, _cfg.GRID_BOTTOM, _cfg.GRID_TOP


# ---------------------------------------------------------------------------
# Geometric vertical grid (fixed for the paper)
# ---------------------------------------------------------------------------

Z_BOUNDARIES  = np.array([0.0, 0.3, 1.2, 4.8, 19.2, 76.8, 200.0])   # layer faces [m]
Z_THICKNESSES = np.diff(Z_BOUNDARIES)                                   # h_k  [m]
Z_CENTERS     = (Z_BOUNDARIES[:-1] + Z_BOUNDARIES[1:]) / 2.0           # cell centres [m]
N_Z           = len(Z_THICKNESSES)                                      # 6

# Distances between consecutive cell centres (used for inter-cell flux)
_Z_DC = np.diff(Z_CENTERS)   # shape (N_Z-1,)


# ---------------------------------------------------------------------------
# Tridiagonal helpers
# ---------------------------------------------------------------------------

def _pack_banded(lower, diag, upper):
    """Pack tridiagonal (lower, diag, upper) into scipy banded (3, N) format."""
    n  = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:]  = upper[:n-1]   # superdiagonal  A[k, k+1]  stored at column k+1
    ab[1, :]   = diag
    ab[2, :n-1] = lower[1:]    # subdiagonal    A[k+1, k]  stored at column k
    return ab


def _horizontal_operator(n, alpha, courant=0.0):
    """
    Implicit horizontal sweep matrix: diffusion (alpha = D dt / dx^2) plus
    first-order upwind advection (courant = u dt / dx, sign = flow direction).

    Both ends are Dirichlet; returns (banded matrix, lo, up) where lo/up are the
    coefficients that multiply the background value on the first/last row's RHS.
    """
    lo = alpha + max(courant, 0.0)     # coupling to the upstream-left neighbour
    up = alpha + max(-courant, 0.0)    # coupling to the upstream-right neighbour
    diag  = np.full(n, 1.0 + lo + up)
    lower = np.full(n, -lo)
    upper = np.full(n, -up)
    return _pack_banded(lower, diag, upper), lo, up


def _make_uniform_banded(n, alpha):
    """(I - alpha * L_uniform) for a pure-diffusion sweep with Dirichlet ends."""
    return _horizontal_operator(n, alpha)[0]


# ---------------------------------------------------------------------------
# Vertical diffusivity on the geometric grid
# ---------------------------------------------------------------------------

def kz_at_faces(Kz=None):
    """
    K_z at the upper face of every layer (0.3, 1.2, ..., 200 m), shape (N_Z,).

    Kz=None    -> Monin-Obukhov profile from atmosphere.py
    Kz=float   -> uniform diffusivity (used by the analytic-comparison tests)
    Kz=ndarray -> taken as given
    """
    if Kz is None:
        return kz_profile(Z_BOUNDARIES[1:])
    if np.ndim(Kz) == 0:
        return np.full(N_Z, float(Kz))
    return np.asarray(Kz, dtype=np.float64)


def _make_z_banded(Kz, dt, lam=0.0):
    """
    Tridiagonal for the implicit z-sweep on the geometric grid.
    - Bottom (k=0) : Neumann (zero flux), no lower neighbour.
    - Interior      : finite-volume centred differences, K_z at the layer faces.
    - Top (k=N_Z-1): Dirichlet (background).  Returns the RHS correction
                      coefficient `rhs_top_coeff` to be multiplied by bg.
    - lam           : first-order loss rate [1/s], backward Euler on the diagonal.
    """
    h, dc = Z_THICKNESSES, _Z_DC
    kf = kz_at_faces(Kz)                       # kf[k] = K at the face above layer k

    lower = np.zeros(N_Z)
    diag  = np.ones(N_Z) + lam * dt
    upper = np.zeros(N_Z)

    # coupling between layer k and k+1 (Neumann bottom: no k=-1 term)
    up_coeff = kf[:-1] * dt / (h[:-1] * dc)            # layer k -> k+1, k = 0..N_Z-2
    dn_coeff = kf[:-1] * dt / (h[1:] * dc)             # layer k+1 -> k

    diag[:-1]  += up_coeff
    upper[:-1]  = -up_coeff
    diag[1:]   += dn_coeff
    lower[1:]   = -dn_coeff

    # Dirichlet top face: distance from centre to face is h/2
    rhs_top = kf[-1] * dt / (h[-1] * (h[-1] / 2.0))
    diag[-1] += rhs_top

    return _pack_banded(lower, diag, upper), rhs_top


# ---------------------------------------------------------------------------
# Single-purpose steps of one timestep
# ---------------------------------------------------------------------------

def inject_emissions(conc, x_vec, y_vec, emission_vec, V0):
    """Add vehicle emissions [mg] to the ground layer of `conc` [mg/m^3], in place."""
    if len(x_vec) == 0:
        return
    NX, NY = conc.shape[:2]
    gl, gr, gb, gt = _grid()
    xi = ((np.asarray(x_vec, dtype=np.float64) - gl) / ((gr - gl) / NX)).astype(np.int32)
    yi = ((np.asarray(y_vec, dtype=np.float64) - gb) / ((gt - gb) / NY)).astype(np.int32)
    ea = np.asarray(emission_vec, dtype=np.float64)
    mask = (xi >= 0) & (xi < NX) & (yi >= 0) & (yi < NY)
    np.add.at(conc[:, :, 0], (xi[mask], yi[mask]), ea[mask] / V0)


def horizontal_sweep(conc, axis, D, velocity, cell_len, dt, bg):
    """Implicit diffusion + upwind advection along `axis` (0 = x, 1 = y), Dirichlet ends."""
    n = conc.shape[axis]
    ab, lo, up = _horizontal_operator(n, D * dt / cell_len**2, velocity * dt / cell_len)
    moved = np.moveaxis(conc, axis, 0)
    rhs = np.ascontiguousarray(moved).reshape(n, -1).copy()
    rhs[0, :]  += lo * bg
    rhs[-1, :] += up * bg
    solved = solve_banded((1, 1), ab, rhs).reshape(moved.shape)
    return np.moveaxis(solved, 0, axis)


def vertical_sweep(conc, Kz, lam, dt, bg):
    """Implicit vertical diffusion with loss of the excess over background, Neumann bottom."""
    ab, top_coeff = _make_z_banded(Kz, dt, lam)
    rhs = np.ascontiguousarray(np.moveaxis(conc, 2, 0)).reshape(conc.shape[2], -1).copy()
    rhs[-1, :] += top_coeff * bg
    rhs += lam * dt * bg          # the loss acts on (C - bg), not on the ambient level
    solved = solve_banded((1, 1), ab, rhs).reshape(np.moveaxis(conc, 2, 0).shape)
    return np.moveaxis(solved, 0, 2)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def process_gas_step(
    conc_mgcell,
    x_vec, y_vec,
    emission_vec,
    cell_len_x_m,
    cell_len_y_m,
    default_value,
    Dx, Dy, Kz=None,
    dt=1.0,
    lam=0.0,
    wind=(0.0, 0.0),
):
    """
    One timestep of 3D pollutant transport (implicit ADI, Lie splitting).

    Parameters
    ----------
    conc_mgcell   : ndarray (NX, NY, NZ)  - mass per cell [mg/cell]
    x_vec, y_vec  : array-like            - vehicle SUMO coordinates
    emission_vec  : array-like            - vehicle emissions [mg] this step
    cell_len_x_m  : float                 - cell width in x [m]
    cell_len_y_m  : float                 - cell width in y [m]
    default_value : float                 - background concentration [mg/m^3]
    Dx, Dy        : float                 - horizontal diffusion coefficients [m^2/s]
    Kz            : None | float | array  - vertical diffusivity; None = Monin-Obukhov
    dt            : float                 - timestep [s]
    lam           : float                 - first-order loss rate of the excess [1/s]
    wind          : (u_x, u_y)            - uniform transport velocity [m/s] (calm by default)

    Returns
    -------
    Updated conc_mgcell [mg/cell], same shape as input.
    """
    V = cell_len_x_m * cell_len_y_m * Z_THICKNESSES
    conc = conc_mgcell / V[np.newaxis, np.newaxis, :]      # mg/cell -> mg/m^3

    inject_emissions(conc, x_vec, y_vec, emission_vec, V[0])
    np.clip(conc, default_value, None, out=conc)

    conc = horizontal_sweep(conc, 0, Dx, wind[0], cell_len_x_m, dt, default_value)
    conc = horizontal_sweep(conc, 1, Dy, wind[1], cell_len_y_m, dt, default_value)
    conc = vertical_sweep(conc, Kz, lam, dt, default_value)

    np.clip(conc, default_value, None, out=conc)
    return conc * V[np.newaxis, np.newaxis, :]


def calculate_optimal_dt(cell_len_x_m=None, cell_len_y_m=None, Dx=None, Dy=None, Dz=None):
    """
    The implicit ADI scheme is unconditionally stable.
    Returns 1.0 s to keep microsimulation source injection realistic.
    """
    return 1.0


def get_emissions_batched(vehicle_ids, dt):
    """
    Retrieve positions and pollutant emissions for all vehicles via TraCI.

    Returns
    -------
    x_vec, y_vec : SUMO coordinate lists
    co_vec, nox_vec, pmx_vec : emission lists [mg] for this timestep
    """
    import traci   # imported here so the solver can be used (and tested) without SUMO installed

    x_vec, y_vec, co_vec, nox_vec, pmx_vec = [], [], [], [], []

    for vid in vehicle_ids:
        try:
            x, y = traci.vehicle.getPosition(vid)
            x_vec.append(x)
            y_vec.append(y)
            co_vec.append(traci.vehicle.getCOEmission(vid)  * dt)
            nox_vec.append(traci.vehicle.getNOxEmission(vid) * dt)
            pmx_vec.append(traci.vehicle.getPMxEmission(vid) * dt)
        except traci.TraCIException:
            pass

    return x_vec, y_vec, co_vec, nox_vec, pmx_vec
