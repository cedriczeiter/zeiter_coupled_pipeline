"""
Vertical turbulence for the dispersion model: Monin-Obukhov similarity theory.

    K_z(z) = kappa * u_* * z / phi_h(z / L)

The inputs are fixed scenario-level values matched to Zhao et al. (2024) and
are deliberately not tuned:  L = -200 m (unstable), z_0 = 0.1 m,
z_ref = 5 m, u_* = 0.91 m/s.  z_0 and z_ref enter only the derivation of u_*
from the 10 m/s reference wind (see friction_velocity); K_z(z) itself depends
on u_*, L and kappa only, and no boundary-layer height z_i is needed.
"""

import numpy as np

KAPPA         = 0.35      # von Karman constant (Businger et al. 1971)
PHI_H_NEUTRAL = 0.74      # turbulent Prandtl number at neutral stability
OBUKHOV_L     = -200.0    # Obukhov length [m], negative = unstable
Z0            = 0.1       # roughness length [m]
Z_REF         = 5.0       # wind reference height [m]
U_STAR        = 0.91      # friction velocity [m/s]
U_REF         = 10.0      # Zhao et al. wind speed at z_ref [m/s]


def phi_h(zeta):
    """Stability function for heat/scalars, phi_h(zeta) with zeta = z / L."""
    zeta = np.asarray(zeta, dtype=np.float64)
    unstable = PHI_H_NEUTRAL * (1.0 - 9.0 * np.minimum(zeta, 0.0)) ** -0.5
    stable   = PHI_H_NEUTRAL + 4.7 * np.maximum(zeta, 0.0)
    return np.where(zeta < 0.0, unstable, stable)


def kz_profile(z, u_star=U_STAR, obukhov_l=OBUKHOV_L, kappa=KAPPA):
    """Vertical eddy diffusivity K_z(z) [m^2/s] at heights z [m] (z > 0)."""
    z = np.asarray(z, dtype=np.float64)
    return kappa * u_star * z / phi_h(z / obukhov_l)


def psi_m(zeta):
    """Businger-Dyer integrated stability correction for momentum."""
    zeta = np.asarray(zeta, dtype=np.float64)
    x = (1.0 - 16.0 * np.minimum(zeta, 0.0)) ** 0.25
    unstable = (2.0 * np.log((1.0 + x) / 2.0) + np.log((1.0 + x**2) / 2.0)
                - 2.0 * np.arctan(x) + np.pi / 2.0)
    stable = -5.0 * np.maximum(zeta, 0.0)
    return np.where(zeta < 0.0, unstable, stable)


def friction_velocity(u_ref=U_REF, z_ref=Z_REF, z0=Z0, obukhov_l=OBUKHOV_L, kappa=KAPPA):
    """u_* from the stability-corrected log wind profile (documents U_STAR)."""
    return float(kappa * u_ref / (np.log(z_ref / z0) - psi_m(z_ref / obukhov_l)))
