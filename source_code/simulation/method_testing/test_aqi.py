"""
Tests for the AQI evaluation (process_annoyance.py) and the time averages (process_data.py).
Standalone (no SUMO):   cd simulation/method_testing && python -m pytest test_aqi.py -q
"""
import numpy as np
import pandas as pd
import pytest

import process_annoyance as pa
from process_annoyance import calculate_aqi_for_pollutant_US_AQI as us_aqi
from process_annoyance import calculate_aqi_for_pollutant_AQIH as aqih
from process_data import process_data_statistics


def test_us_aqi_band_edges():
    assert us_aqi(0.0, 'CO') == 0
    assert us_aqi(4.4, 'CO') == pytest.approx(50)
    assert us_aqi(9.4, 'CO') == pytest.approx(100)
    assert us_aqi(9.0, 'PM2.5') == pytest.approx(50)     # EPA 2024 revision
    assert us_aqi(1e6, 'NO2') == 500


def test_us_aqi_is_continuous_and_monotone_across_band_gaps():
    for pol, top in (('CO', 50), ('NO2', 2100), ('PM2.5', 330)):
        c = np.linspace(0, top, 20001)
        aqi = np.array([us_aqi(x, pol) for x in c])
        assert np.all(np.diff(aqi) >= -1e-9)
        assert np.max(np.abs(np.diff(aqi))) < 1.0   # no jumps (e.g. between 4.4 and 4.5 ppm)


def test_aqih_range_and_monotone():
    c = np.linspace(0, 800, 2000)
    for pol in ('NO2', 'PM2.5'):
        v = np.array([aqih(x, pol) for x in c])
        assert v.min() == 1 and v.max() == 10 and np.all(np.diff(v) >= 0)


def test_aqih_has_no_co():
    with pytest.raises(AssertionError):
        aqih(1.0, 'CO')


def test_unit_conversions():
    assert pa.POLLUTANT_CONFIG['co']['conversion_factor_us_AQI'] == pytest.approx(24.45 / 28.01)
    assert pa.POLLUTANT_CONFIG['nox']['conversion_factor_us_AQI'] == pytest.approx(24.45 / 46.01 * 1000)


def test_statistics_and_aqi_pipeline_on_a_small_grid(tmp_path):
    """Grid size comes from the data, the 1h NO2 maximum is the per-cell maximum over the hours."""
    data, avg, fin = tmp_path / "data", tmp_path / "avg", tmp_path / "final"
    for d in (data, avg, fin):
        d.mkdir()
    rng = np.random.default_rng(0)
    nox = {h: 0.0051 + rng.random((7, 5)) * 0.2 for h in (1, 2, 3)}
    for h, arr in nox.items():
        for gas in ("co", "nox", "pmx"):
            pd.DataFrame(arr).to_csv(data / f"data_{gas}_1_{h * 3600}.0.csv", index=False)
        pd.DataFrame(arr * 0).to_csv(data / f"data_nox_0_{h * 3600}.0.csv", index=False)   # other layer: ignored
    (data / "data_junk.csv").write_text("1,2\n")   # a stray file must not stop the processing

    process_data_statistics(str(data), avg_data_folder=str(avg))
    pa.process_data_annoyance(str(data), str(fin), str(avg))

    expected = np.max([[[us_aqi(c * pa.POLLUTANT_CONFIG['nox']['conversion_factor_us_AQI'], 'NO2')
                         for c in row] for row in arr] for arr in nox.values()], axis=0)
    got = pd.read_csv(fin / "nox_max_aqi.csv").values
    assert got.shape == (7, 5) and np.allclose(got, expected)
    assert (fin / "overall_max_aqi.csv").exists() and (fin / "overall_max_aqih.csv").exists()
    assert not (fin / "pmx_max_aqi.csv").exists()   # 24 h average needs 24 snapshots
