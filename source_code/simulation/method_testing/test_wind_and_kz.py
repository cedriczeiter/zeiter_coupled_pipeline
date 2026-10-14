"""
Tests for the Monin-Obukhov K_z(z), the PM loss term and the optional wind.
Standalone (no SUMO):   cd simulation/method_testing && python -m pytest test_wind_and_kz.py -q
"""

import numpy as np
import pytest

import atmosphere as atm
import emission_models as em
from emission_models import process_gas_step, Z_THICKNESSES, N_Z
from wind import WindSchedule, wind_components, parse_wind_spec

NX, NY, DX = 61, 61, 10.0
BG = 0.0


@pytest.fixture(autouse=True)
def grid(monkeypatch):
    monkeypatch.setattr(em, "_grid", lambda: (0.0, NX * DX, 0.0, NY * DX))


def _empty(bg=BG):
    return np.full((NX, NY, N_Z), bg) * (DX * DX * Z_THICKNESSES)[None, None, :]


def _step(c, wind=(0.0, 0.0), n=1, emis=(), lam=0.0, Kz=None, Dx=2.0, bg=BG):
    xs = [NX * DX / 2] * len(emis)
    for i in range(n):
        e = emis if i == 0 else ()
        c = process_gas_step(c, xs if i == 0 else [], xs if i == 0 else [], list(e),
                             DX, DX, bg, Dx, Dx, Kz, 1.0, lam=lam, wind=wind)
    return c


def _centroid(c):
    ground = c[:, :, 0]
    xs, ys = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    return (ground * xs).sum() / ground.sum(), (ground * ys).sum() / ground.sum()


# ---------------------------------------------------------------- Monin-Obukhov

def test_fixed_inputs_match_zhao():
    assert atm.OBUKHOV_L == -200.0 and atm.Z0 == 0.1 and atm.Z_REF == 5.0
    assert atm.U_STAR == 0.91


def test_ustar_consistent_with_log_profile():
    assert atm.friction_velocity() == pytest.approx(atm.U_STAR, abs=0.01)


def test_phi_h_limits_and_continuity():
    assert atm.phi_h(0.0) == pytest.approx(0.74)
    assert atm.phi_h(-1.0) == pytest.approx(0.74 / np.sqrt(10.0))
    assert atm.phi_h(1e-9) == pytest.approx(atm.phi_h(-1e-9), abs=1e-6)
    assert atm.phi_h(1.0) == pytest.approx(0.74 + 4.7)


def test_kz_positive_increasing_and_needs_no_boundary_layer_height():
    k = atm.kz_profile(em.Z_BOUNDARIES[1:])
    assert np.all(k > 0) and np.all(np.diff(k) > 0)
    assert k[0] == pytest.approx(0.35 * 0.91 * 0.3 / atm.phi_h(0.3 / -200.0))
    import inspect
    assert "z_i" not in inspect.signature(atm.kz_profile).parameters


def test_default_step_uses_monin_obukhov():
    c = _step(_empty(), emis=[100.0], n=5)
    c_ref = _step(_empty(), emis=[100.0], n=5, Kz=atm.kz_profile(em.Z_BOUNDARIES[1:]))
    assert np.array_equal(c, c_ref)


# ------------------------------------------------------------------- loss term

def test_loss_removes_excess_at_the_backward_euler_rate():
    bg = 0.01
    c = _empty(bg)
    c[NX // 2, NY // 2, 1] += 5.0
    excess0 = (c - _empty(bg)).sum()
    out = _step(c, Kz=0.0, Dx=0.0, lam=0.01, n=50, bg=bg)
    assert (out - _empty(bg)).sum() == pytest.approx(excess0 * (1.01) ** -50, rel=1e-9)


def test_zero_loss_keeps_mass_without_boundary_contact():
    c = _step(_empty(), emis=[100.0], n=3, Dx=1.0, Kz=0.0)
    assert c.sum() == pytest.approx(100.0, rel=1e-9)


# ------------------------------------------------------------------------ wind

def test_wind_components_follow_compass_convention():
    assert wind_components(0, 5) == (0.0, -5.0)         # from north -> blows south
    assert wind_components(90, 5) == pytest.approx((-5.0, 0.0))
    assert wind_components(180, 5) == (0.0, 5.0)
    assert wind_components(270, 5) == pytest.approx((5.0, 0.0))
    ux, uy = wind_components(315, 10)                   # from north-west -> towards south-east
    assert ux > 0 and uy < 0 and np.hypot(ux, uy) == pytest.approx(10.0)


def test_schedule_is_calm_by_default_and_per_hour():
    assert not WindSchedule().active and WindSchedule().at(12 * 3600) == (0.0, 0.0)
    w = parse_wind_spec("8-9=270:4;18=0:2")
    assert w.at(7.9 * 3600) == (0.0, 0.0)
    assert w.at(8.5 * 3600)[0] == pytest.approx(4.0)
    assert w.at(9 * 3600 + 59)[0] == pytest.approx(4.0)
    assert w.at(10 * 3600) == (0.0, 0.0)
    assert w.at(18 * 3600)[1] == pytest.approx(-2.0)
    assert parse_wind_spec("315:10").at(3 * 3600) == wind_components(315, 10)
    assert not parse_wind_spec(None).active


@pytest.mark.parametrize("spec", ["abc", "8=315", "25=1:1", "5=1:-2"])
def test_bad_wind_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        parse_wind_spec(spec)


def test_calm_wind_is_identical_to_no_wind_argument():
    a = _step(_empty(), emis=[50.0], n=10)
    b = _step(_empty(), emis=[50.0], n=10, wind=(0.0, 0.0))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("direction, sx, sy", [(270, +1, 0), (90, -1, 0), (180, 0, +1), (0, 0, -1)])
def test_plume_moves_downwind(direction, sx, sy):
    u = wind_components(direction, 3.0)
    cx0, cy0 = _centroid(_step(_empty(), emis=[100.0], n=1, Dx=1.0))
    cx, cy = _centroid(_step(_empty(), emis=[100.0], n=20, wind=u, Dx=1.0))
    assert np.sign(round(cx - cx0, 3)) == sx and np.sign(round(cy - cy0, 3)) == sy


def test_plume_displacement_matches_speed():
    n, speed = 20, 2.0
    cx0, _ = _centroid(_step(_empty(), emis=[100.0], n=1, Dx=1.0))
    cx, _ = _centroid(_step(_empty(), emis=[100.0], n=n, wind=(speed, 0.0), Dx=1.0))
    assert (cx - cx0) * DX == pytest.approx(speed * n, rel=1e-6)


def test_wind_conserves_mass_until_the_plume_reaches_the_boundary():
    c = _step(_empty(), emis=[100.0], n=15, wind=(3.0, 1.5), Dx=1.0, Kz=0.0)
    assert c.sum() == pytest.approx(100.0, rel=1e-9)


def test_inflow_boundary_stays_at_background_and_field_stays_stable():
    bg = 0.02
    c = _step(_empty(bg), emis=[500.0], n=60, wind=wind_components(315, 20), bg=bg)
    assert np.isfinite(c).all() and c.min() >= bg * DX * DX * Z_THICKNESSES.min() * 0.999
    conc = c / (DX * DX * Z_THICKNESSES)[None, None, :]
    assert conc[0, :, 0].max() < 0.5 and conc.min() >= bg - 1e-12


def test_stronger_wind_transports_further():
    c_slow = _centroid(_step(_empty(), emis=[100.0], n=20, wind=(1.0, 0.0), Dx=1.0))[0]
    c_fast = _centroid(_step(_empty(), emis=[100.0], n=20, wind=(5.0, 0.0), Dx=1.0))[0]
    assert c_fast > c_slow


def test_wind_does_not_change_vertical_turbulence():
    """K_z is independent of the wind: with no horizontal gradients the vertical profile is unchanged."""
    c0 = _empty(0.01) + 0.0
    c0[:, :, 0] += 5.0            # horizontally uniform excess in the lowest layer
    a = _step(c0.copy(), n=5, bg=0.01)
    b = _step(c0.copy(), n=5, bg=0.01, wind=(6.0, 6.0))
    assert np.allclose(a[NX // 2, NY // 2], b[NX // 2, NY // 2], rtol=1e-6)


def test_injection_lands_in_correct_cell_on_nonsquare_grid(monkeypatch):
    NX, NY, dx, dy = 30, 20, 4.0, 6.0
    monkeypatch.setattr(em, "_grid", lambda: (0.0, NX * dx, 0.0, NY * dy))
    c = process_gas_step(np.zeros((NX, NY, N_Z)), [4.0 * 7.5], [6.0 * 3.5], [100.0],
                         dx, dy, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert np.unravel_index(np.argmax(c[:, :, 0]), (NX, NY)) == (7, 3)


def test_out_of_domain_emissions_are_ignored(monkeypatch):
    NX = NY = 10
    monkeypatch.setattr(em, "_grid", lambda: (0.0, 50.0, 0.0, 50.0))
    c = process_gas_step(np.zeros((NX, NY, N_Z)), [-5.0, 999.0], [10.0, 10.0], [50.0, 50.0],
                         5.0, 5.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert c.sum() == 0.0
