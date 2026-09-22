"""vol_eval.py icin birim testleri (QLIKE, MSE, R^2, Mincer-Zarnowitz, DM testi,
pinball loss, kapsama)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vol_eval import (  # noqa: E402
    diebold_mariano,
    interval_coverage,
    mincer_zarnowitz,
    mse,
    pinball_loss,
    qlike,
    qlike_series,
    r_squared_log,
)


def test_qlike_zero_for_perfect_forecast():
    actual = [1.0, 2.0, 0.5, 3.0]
    assert qlike(actual, actual) == pytest.approx(0.0, abs=1e-9)


def test_qlike_positive_for_imperfect_forecast():
    assert qlike([1.0, 2.0], [2.0, 1.0]) > 0


def test_qlike_none_when_no_valid_pairs():
    assert qlike([-1.0], [-1.0]) is None


def test_mse_zero_for_perfect_forecast():
    assert mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_mse_matches_manual_calculation():
    assert mse([1.0, 2.0], [2.0, 4.0]) == pytest.approx((1 + 4) / 2)


def test_r_squared_log_perfect_is_one():
    vals = [0.1, 0.5, -0.2, 0.8]
    assert r_squared_log(vals, vals) == pytest.approx(1.0, abs=1e-9)


def test_r_squared_log_constant_predictor_is_negative_or_zero():
    actual = [0.1, 0.5, -0.2, 0.8, 0.3]
    const_pred = [np.mean(actual)] * len(actual)
    assert r_squared_log(actual, const_pred) <= 1e-9


def test_mincer_zarnowitz_perfect_forecast_has_intercept_zero_slope_one():
    # Tam esitlikte (actual==predicted) kalinti varyansi tam sifir olur ve
    # F-testi sayisal olarak yozlasir (0/0) - bu yuzden a=0,b=1 tam dogru
    # olsa bile pvalue anlamsizlasir. Testi yozlasmadan kacinmak icin kucuk
    # gurultu eklenmis, tarafsiz (unbiased) bir tahminciyle kontrol ediyoruz.
    # actual = predicted + BAGIMSIZ gurultu (predicted = actual + gurultu DEGIL):
    # boylece populasyonda E[actual|predicted]=predicted (a=0,b=1) TAM saglanir,
    # OLS kalintilari predicted'dan bagimsizdir ve test guvenilir sekilde
    # "iyi kalibre" sonucunu vermelidir.
    rng = np.random.default_rng(0)
    predicted = rng.normal(0, 1, 500)
    actual = predicted + rng.normal(0, 0.05, 500)
    mz = mincer_zarnowitz(actual, predicted)
    assert mz["intercept"] == pytest.approx(0.0, abs=0.05)
    assert mz["slope"] == pytest.approx(1.0, abs=0.05)
    assert mz["well_calibrated"] is True


def test_mincer_zarnowitz_too_few_points_returns_none():
    mz = mincer_zarnowitz([1.0, 2.0], [1.0, 2.0])
    assert mz["intercept"] is None
    assert mz["well_calibrated"] is None


def test_diebold_mariano_prefers_lower_loss_method():
    rng = np.random.default_rng(1)
    loss_a = rng.normal(0.1, 0.05, 300)  # dusuk kayip
    loss_b = rng.normal(0.5, 0.05, 300)  # yuksek kayip
    dm = diebold_mariano(loss_a, loss_b)
    assert dm["significant"] is True
    assert dm["a_better"] is True


def test_diebold_mariano_no_difference_when_identical():
    rng = np.random.default_rng(2)
    loss = rng.normal(0.2, 0.05, 300)
    dm = diebold_mariano(loss, loss)
    assert dm["dm_stat"] == pytest.approx(0.0, abs=1e-6)


def test_pinball_loss_zero_for_exact_match():
    assert pinball_loss([1.0, 2.0], [1.0, 2.0], 0.5) == pytest.approx(0.0)


def test_pinball_loss_asymmetric_for_extreme_quantile():
    # tau=0.9: gercek deger tahminin ALTINDA kalirsa (asiri tahmin) daha az cezalandirilir
    over_loss = pinball_loss([1.0], [2.0], 0.9)  # tahmin gercekten yuksek
    under_loss = pinball_loss([2.0], [1.0], 0.9)  # tahmin gercekten dusuk
    assert under_loss > over_loss


def test_interval_coverage_full_when_always_inside():
    assert interval_coverage([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]) == 100.0


def test_interval_coverage_zero_when_always_outside():
    assert interval_coverage([100.0, 200.0], [0.0, 0.0], [1.0, 1.0]) == 0.0


def test_interval_coverage_partial():
    actual = [1.0, 5.0, 1.0, 5.0]
    lower = [0.0] * 4
    upper = [2.0] * 4
    assert interval_coverage(actual, lower, upper) == pytest.approx(50.0)
